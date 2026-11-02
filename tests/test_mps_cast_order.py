"""Tripwire for the cast orders Metal gets wrong, and for what depends on them.

torch 2.14: naming a device and a 64-bit dtype in one call on an MPS tensor returns
**zeros** without raising, and casting to float64 before leaving Metal raises. Both
read as perfectly good code on the CUDA box where they are written, so they are
enforced here rather than remembered. See "MPS traps in torch 2.14" in the wiki.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import fastfuncstuff
from fastfuncstuff import policy_linalg as plinalg
from fastfuncstuff.cli_utils import parse_device_arg, setup_device
from fastfuncstuff.glm.arma import _ljung_box_batched
from fastfuncstuff.utils import MPS_DEVICE, get_device

_F64 = {"torch.float64", "torch.double", "torch.complex128", "torch.cdouble"}
_F64_TYPES = {"torch.DoubleTensor", "torch.cuda.DoubleTensor"}
_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Metal")


def _violations(source: str) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords}
        args = [ast.unparse(a) for a in node.args]
        if node.func.attr == "to":
            combined = (kwargs.get("dtype") in _F64 and ("device" in kwargs or args)) or (
                len(args) == 2 and args[1] in _F64
            )
            if combined:
                out.append((node.lineno, "device and float64 in one .to()"))
        if node.func.attr == "type" and args and args[0] in _F64 | _F64_TYPES:
            out.append((node.lineno, "float64 through .type()"))
        moving_off = node.func.attr == "cpu"
        receiver = node.func.value
        if moving_off and isinstance(receiver, ast.Call):
            inner = ast.unparse(receiver.func)
            casted = inner.endswith(".double") or (
                inner.endswith(".to")
                and [ast.unparse(a) for a in receiver.args] in ([f] for f in _F64)
            )
            if casted:
                out.append((node.lineno, "float64 cast before .cpu()"))
    return out


def test_detector_catches_each_form():
    """The scan below is only worth anything if these keep being caught."""
    for bad in (
        "x.to('cpu', torch.float64)",
        "x.to(device=d, dtype=torch.float64)",
        "x.to(dtype=torch.double, device=d)",
        "x.to(d, dtype=torch.float64)",
        "x.to('cpu', torch.complex128)",
        "x.type(torch.DoubleTensor)",
        "x.type(torch.float64)",
        "x.double().cpu()",
        "x.to(torch.float64).cpu()",
    ):
        assert _violations(bad), bad
    for good in (
        "x.to(device=d).to(torch.float64)",
        "x.cpu().double()",
        "x.to(d, torch.float32)",
        "x.to(d, dtype=torch.float32)",
        "s.dtype.type(0.5)",
    ):
        assert not _violations(good), good


def test_package_has_no_unsafe_float64_casts():
    root = Path(fastfuncstuff.__file__).parent
    found = [
        f"{path.relative_to(root)}:{line}: {what}"
        for path in sorted(root.rglob("*.py"))
        for line, what in _violations(path.read_text())
    ]
    assert not found, "\n".join(found)


_ROUTED_NAMES = {
    "qr", "svd", "svdvals", "matrix_rank", "cond", "pinv", "lstsq", "solve", "solve_triangular",
    "cholesky", "cholesky_ex", "inv", "inv_ex", "det", "slogdet", "lu_factor", "lu_solve", "eigh",
    "eigvalsh",
}  # fmt: skip
# The second row is torch's older spellings of the same solvers: nothing routes them,
# and nothing here calls them yet.
_ROUTED_TOP_LEVEL = {
    "cholesky_solve", "cholesky_inverse", "logdet",
    "svd", "qr", "inverse", "pinverse", "triangular_solve", "det", "slogdet", "lu", "cholesky",
}  # fmt: skip


def test_package_uses_routed_solvers():
    """Every bare solver is a QR waiting to take 35 ms per call on a Mac. Its own module excepted."""
    root = Path(fastfuncstuff.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.name in ("policy_linalg.py", "utils.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in _ROUTED_NAMES | _ROUTED_TOP_LEVEL:
                text = ast.unparse(node)
                if text in {f"torch.linalg.{node.attr}", f"_torch.linalg.{node.attr}"} or (
                    text in {f"torch.{n}" for n in _ROUTED_TOP_LEVEL}
                ):
                    found.append(f"{path.relative_to(root)}:{node.lineno}: {text}")
    assert not found, "use fastfuncstuff.policy_linalg:\n" + "\n".join(found)


def test_routed_solvers_return_what_torch_returns_on_the_cpu():
    a = torch.randn(30, 4, generator=torch.Generator().manual_seed(0))
    q, r = plinalg.qr(a)
    torch.testing.assert_close(q @ r, a, rtol=1e-5, atol=1e-5)
    assert plinalg.cholesky_ex(a.T @ a).info.item() == 0
    assert plinalg.svd(a, full_matrices=False).S.shape == (4,)
    spd = a.T @ a
    inverse = plinalg.cholesky_inverse(plinalg.cholesky(spd))
    torch.testing.assert_close(inverse @ spd, torch.eye(4), rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@_mps
@pytest.mark.parametrize(
    "name,args",
    [
        ("qr", (torch.randn(300, 12),)),
        ("svd", (torch.randn(4, 16, 8),)),  # raises inside Metal's compiler if it stays there
        ("svdvals", (torch.randn(4, 16, 8),)),
        ("inv", (torch.eye(4) * 2,)),
        ("cholesky_inverse", (torch.eye(4) * 2,)),
        ("eigh", (torch.eye(8).expand(3, 8, 8) * 2,)),
    ],
)
def test_routed_solvers_come_home_to_metal(name, args):
    out = getattr(plinalg, name)(*(a.to("mps") for a in args))
    first = out if isinstance(out, torch.Tensor) else out[0]
    assert first.device.type == "mps"


@pytest.mark.gpu
@_mps
def test_routed_solver_takes_a_factor_on_metal_with_a_rhs_on_the_cpu():
    factor = (torch.eye(4) * 2).to("mps")
    x = plinalg.cholesky_solve(torch.ones(4, 1), factor)
    assert x.device.type == "mps"
    torch.testing.assert_close(x.cpu(), torch.full((4, 1), 0.25))


@pytest.mark.gpu
@_mps
def test_fwhmx_detrend_on_metal_matches_cpu():
    """_trigref is float64, so it comes back on the CPU beside residuals still on Metal."""
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    resid = torch.randn(4096, 40, generator=torch.Generator().manual_seed(0))
    mask = torch.ones(16, 16, 16, dtype=torch.bool)
    cpu, metal = (
        estimate_fwhmx_run(
            resid.to(d), mask.to(d), (16, 16, 16), (2.0, 2.0, 2.0), corder=1, device=torch.device(d)
        )
        for d in ("cpu", "mps")
    )
    assert metal.classic_fwhm == pytest.approx(cpu.classic_fwhm, rel=1e-3)


@pytest.mark.gpu
@_mps
def test_simulated_bold_on_metal_runs_and_lands_on_the_cpu():
    """Signal built on Metal plus float64 noise on the CPU used to raise on the sum."""
    from fastfuncstuff.simulation.core import simulate_bold

    out = simulate_bold(
        onsets=[[[10.0, 30.0]]],
        durations=[2.0],
        tr=1.0,
        n_timepoints_per_run=[60],
        amplitude_psc=[1.0],
        tsnr=50.0,
        n_voxels=8,
        device=MPS_DEVICE,
    )
    assert out["data"].device.type == "cpu"


def test_every_device_source_hands_out_the_indexed_mps_device():
    """A tensor reports mps:0, and torch.device("mps") != torch.device("mps:0")."""
    assert MPS_DEVICE == torch.device("mps:0")
    assert parse_device_arg("mps")[0] == MPS_DEVICE


def test_float64_only_tool_runs_on_cpu_when_mps_is_requested(monkeypatch):
    monkeypatch.setattr(
        "fastfuncstuff.cli_utils.parse_device_arg", lambda spec: (MPS_DEVICE, None, None)
    )
    with pytest.warns(UserWarning, match="ffs_test_tool is float64-only"):
        assert setup_device("mps", float64_only="ffs_test_tool").type == "cpu"
    # Without the flag the request stands: Metal is only refused by the tool that needs to.
    monkeypatch.setattr("fastfuncstuff.utils.configure_torch_backends", lambda *a, **k: None)
    assert setup_device("mps") == MPS_DEVICE


@pytest.mark.gpu
@_mps
def test_get_device_compares_equal_to_its_own_tensors():
    assert torch.zeros(1, device=get_device("mps")).device == get_device("mps")


@pytest.mark.gpu
@_mps
def test_ljung_box_on_metal_matches_cpu():
    resid = torch.randn(50, 40, generator=torch.Generator().manual_seed(0))
    got = _ljung_box_batched(resid.to("mps"))
    assert got.dtype == torch.float64
    # Same float32 input and both reductions run on the CPU: identical.
    assert torch.equal(got, _ljung_box_batched(resid))
