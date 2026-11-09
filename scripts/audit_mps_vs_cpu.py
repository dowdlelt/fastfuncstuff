"""Compare MPS against CPU across the torch ops this repo leans on, at sizes that
straddle the power-of-two boundaries where Metal kernels have gone wrong silently.

Why it exists: torch 2.14's replicate pad along depth returned wrong values once the
in-plane size reached 2**16 -- fine in every unit test (tiny volumes), wrong on any
real 256+ voxel brain, and it never raised. Silent failures like that sit at index-
width thresholds, so every op is run just under and just over each one:

    2**16 per plane / per dimension,  2**24 total elements (float32 integer limit),
    and a long single axis.

Usage:  python scripts/audit_mps_vs_cpu.py            # whole sweep
        python scripts/audit_mps_vs_cpu.py pad conv   # only ops whose name matches

Prints one line per (op, shape); ``BAD`` lines are the findings. Re-run after a torch
upgrade and add a ``_MPS_CPU_OPS`` row or a workaround for anything that turns up.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

TOL = 1e-3
GEN = torch.Generator().manual_seed(0)

# (D, H, W) volumes: plane H*W just under / over 2**16, and totals around 2**24.
VOLUMES = [
    (8, 255, 255),
    (8, 256, 256),
    (20, 300, 300),
    (64, 256, 256),
    (255, 255, 255),
    (256, 256, 256),
    (320, 320, 320),
]
# Long single axes around 2**16 and 2**24.
LONG = [65535, 65536, 70000, 2**24 - 1, 2**24 + 1]

# One fixed sampling grid: drawing it inside the case gave CPU and MPS different grids.
_GRID = torch.rand(1, 8, 8, 8, 3, generator=torch.Generator().manual_seed(7)) * 2 - 1

CASES: list[tuple[str, callable, list[tuple[int, ...]]]] = []


def case(name, shapes):
    def deco(fn):
        CASES.append((name, fn, shapes))
        return fn

    return deco


def _vol5(x):
    return x[None, None]


for mode in ("replicate", "reflect", "circular", "constant"):
    for label, pad in [
        ("z", (0, 0, 0, 0, 2, 2)),
        ("y", (0, 0, 2, 2, 0, 0)),
        ("x", (2, 2, 0, 0, 0, 0)),
        ("all", (2,) * 6),
    ]:
        CASES.append(
            (f"pad3d {mode} {label}", lambda x, p=pad, m=mode: F.pad(_vol5(x), p, mode=m), VOLUMES)
        )

for label, k in [("z", (3, 1, 1)), ("y", (1, 3, 1)), ("x", (1, 1, 3)), ("full", (3, 3, 3))]:
    CASES.append(
        (
            f"conv3d {label}",
            lambda x, k=k: F.conv3d(_vol5(x), torch.ones(1, 1, *k, device=x.device) / 3),
            VOLUMES,
        )
    )
CASES.append(
    (
        "conv3d z grouped x3",
        lambda x: F.conv3d(
            torch.stack([x, x * 2, x * 3])[None],
            torch.ones(3, 1, 3, 1, 1, device=x.device) / 3,
            groups=3,
        ),
        VOLUMES,
    )
)
CASES.append(("avg_pool3d", lambda x: F.avg_pool3d(_vol5(x), 2), VOLUMES))
CASES.append(("max_pool3d", lambda x: F.max_pool3d(_vol5(x), 2), VOLUMES))
CASES.append(
    (
        "interpolate trilinear",
        lambda x: F.interpolate(_vol5(x), scale_factor=0.5, mode="trilinear"),
        VOLUMES,
    )
)
CASES.append(
    (
        "interpolate nearest",
        lambda x: F.interpolate(_vol5(x), scale_factor=0.5, mode="nearest"),
        VOLUMES,
    )
)
for d in range(3):
    CASES.append((f"cumsum dim{d}", lambda x, d=d: torch.cumsum(x, d), VOLUMES))
    CASES.append((f"sum dim{d}", lambda x, d=d: x.sum(d), VOLUMES))
    CASES.append((f"flip dim{d}", lambda x, d=d: x.flip(d), VOLUMES))
    CASES.append((f"roll dim{d}", lambda x, d=d: x.roll(3, d), VOLUMES))
CASES.append(("sum all", lambda x: x.sum(), VOLUMES))
CASES.append(("mean all", lambda x: x.mean(), VOLUMES))
CASES.append(("std all", lambda x: x.std(), VOLUMES))
CASES.append(("nonzero count", lambda x: (x > 0.5).nonzero().shape[0] + 0.0 * x.sum(), VOLUMES))
CASES.append(("masked_select sum", lambda x: x[x > 0.5].sum(), VOLUMES))
CASES.append(
    (
        "unfold z + matmul",
        lambda x: torch.matmul(
            F.pad(x, (0, 0, 0, 0, 1, 1)).unfold(0, 3, 1), torch.ones(3, device=x.device)
        ),
        VOLUMES,
    )
)
CASES.append(
    (
        "index_select z",
        lambda x: x.index_select(0, torch.arange(x.shape[0] - 1, device=x.device)),
        VOLUMES,
    )
)
CASES.append(
    (
        "index_add flat",
        lambda x: torch.zeros(1000, device=x.device).index_add(
            0, (torch.arange(x.numel(), device=x.device) % 1000), x.reshape(-1)
        ),
        VOLUMES,
    )
)
CASES.append(
    (
        "grid_sample 3d",
        lambda x: F.grid_sample(
            _vol5(x),
            _GRID.to(x.device),
            align_corners=True,
        ),
        VOLUMES,
    )
)
CASES.append(
    (
        "conv1d long",
        lambda x: F.conv1d(x[None, None], torch.ones(1, 1, 3, device=x.device) / 3, padding=1),
        [(n,) for n in LONG],
    )
)
CASES.append(
    (
        "pad1d replicate long",
        lambda x: F.pad(x[None, None], (2, 2), mode="replicate"),
        [(n,) for n in LONG],
    )
)
CASES.append(("cumsum long", lambda x: torch.cumsum(x, 0), [(n,) for n in LONG]))
CASES.append(("sum long", lambda x: x.sum(), [(n,) for n in LONG]))
CASES.append(
    ("sort long", lambda x: x.sort().values[:: max(1, x.numel() // 997)], [(n,) for n in LONG])
)
CASES.append(("topk long", lambda x: x.topk(5).values, [(n,) for n in LONG]))
CASES.append(
    (
        "conv2d 2d big",
        lambda x: F.conv2d(x[None, None], torch.ones(1, 1, 3, 3, device=x.device) / 9, padding=1),
        [(255, 255), (256, 256), (1024, 1024), (4097, 4097)],
    )
)
CASES.append(
    (
        "pad2d replicate big",
        lambda x: F.pad(x[None, None], (2, 2, 2, 2), mode="replicate"),
        [(255, 255), (256, 256), (1024, 1024), (4097, 4097)],
    )
)


def _canary_ok() -> bool:
    """A tiny known op on MPS. After a Metal fault the device returns zeros for
    everything, which would make every later case look like a finding."""
    try:
        x = torch.arange(1.0, 101.0, device="mps")
        return float(x.sum().cpu()) == 5050.0
    except Exception:  # noqa: BLE001
        return False


def run_case(index: int) -> int:
    name, fn, shapes = CASES[index]
    bad = 0
    for shape in shapes:
        x = torch.rand(*shape, generator=GEN)
        try:
            ref = fn(x)
            got = fn(x.to("mps"))
            ref = torch.as_tensor(ref, dtype=torch.float32)
            got = torch.as_tensor(got).cpu().to(torch.float32)
            if not _canary_ok():
                print(
                    f"{name:26s} {str(shape):18s} GPU-FAULT (canary failed; later shapes skipped)",
                    flush=True,
                )
                return 2
            if ref.shape != got.shape:
                verdict = f"BAD shape {tuple(got.shape)} != {tuple(ref.shape)}"
            else:
                scale = max(float(ref.abs().max()), 1.0)
                err = float((ref - got).abs().max()) / scale
                verdict = f"BAD rel-err {err:.3g}" if err > TOL else "ok"
        except Exception as exc:  # noqa: BLE001 -- an exception is itself a finding
            verdict = f"ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
        bad += not verdict.startswith("ok")
        print(f"{name:26s} {str(shape):18s} {verdict}", flush=True)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return 0
    if argv[:1] == ["--case"]:
        return run_case(int(argv[1]))
    import subprocess

    problems = 0
    for i, (name, _, _) in enumerate(CASES):
        if argv and not any(f in name for f in argv):
            continue
        # One process per op: a Metal fault poisons the whole process, so isolating
        # them keeps one broken op from turning every later line into a false BAD.
        r = subprocess.run(
            [sys.executable, "-I", __file__, "--case", str(i)], capture_output=True, text=True
        )
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        print(
            "\n".join(lines) if lines else f"{name:26s} (no output; rc={r.returncode})", flush=True
        )
        problems += sum(not ln.rstrip().endswith(" ok") for ln in lines)
    print(f"\n{problems} problem line(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
