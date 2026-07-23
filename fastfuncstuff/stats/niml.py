"""
3dClustSim-style NIML cluster-table writer and 3drefit header injection.

The cluster table consists of one NIML element per ``(NN, sidedness)``
combination::

    <3dClustSim_NN1
      commandline="..."
      thresholding="1-sided"
      nxyz="X,Y,Z"
      dxyz="dx,dy,dz"
      fwhmxyz="0,0,0"     # always 0 — our null is from permutations, not smoothing
      iter="<n_perms>"
      pthr="0.05,0.02,..."
      athr="0.10,0.05,..."
      mask_count="<count>" >
       <floats: one column per athr, one row per pthr>
    </3dClustSim_NN1>

The element is attached to the stat dataset's AFNI extension via
``3drefit -atrstring AFNI_CLUSTSIM_NN{n}_{1sided|2sided} file:...``.

A base64-encoded byte mask is written and attached as
``AFNI_CLUSTSIM_MASK`` so AFNI knows which voxels were eligible.
"""

from __future__ import annotations

import base64
import sys
import textwrap
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# NIML serialisation
# ---------------------------------------------------------------------------


def write_clustsim_niml(
    out_path: Path,
    table: np.ndarray,
    *,
    nn: int,
    sidedness: str,
    commandline: str,
    nxyz: tuple[int, int, int],
    dxyz: tuple[float, float, float],
    pthr: tuple[float, ...],
    athr: tuple[float, ...],
    n_perms: int,
    mask_count: int | None = None,
    mask_idcode: str | None = None,
    mask_name: str | None = None,
) -> None:
    """Write a single ``3dClustSim_NN{n}`` NIML element to ``out_path``.

    ``table`` has shape ``[len(pthr), len(athr)]``.  The NIML format writes
    one float column per athr (the second index), with rows indexed by pthr.
    """
    if table.shape != (len(pthr), len(athr)):
        raise ValueError(f"table shape {table.shape} != (npthr={len(pthr)}, nathr={len(athr)})")

    pthr_csv = ",".join(f"{p:.6f}" for p in pthr)
    athr_csv = ",".join(f"{a:.6f}" for a in athr)
    nxyz_csv = ",".join(str(x) for x in nxyz)
    dxyz_csv = ",".join(f"{d:.4f}" for d in dxyz)

    lines = [
        f"<3dClustSim_NN{nn}",
        f'  ni_type="{len(athr)}*float"',
        f'  ni_dimen="{len(pthr)}"',
        f'  commandline="{commandline}"',
        f'  thresholding="{sidedness}"',
        f'  nxyz="{nxyz_csv}"',
        f'  dxyz="{dxyz_csv}"',
        '  fwhmxyz="0,0,0"',
        f'  iter="{n_perms}"',
        f'  pthr="{pthr_csv}"',
        f'  athr="{athr_csv}"',
    ]
    if mask_idcode is not None:
        lines.append(f'  mask_dset_idcode="{mask_idcode}"')
    if mask_name is not None:
        lines.append(f'  mask_dset_name="{mask_name}"')
    # AFNI emits the closing `>` on the same line as the last attribute
    # (typically mask_count).  Match that.
    if mask_count is not None:
        lines.append(f'  mask_count="{mask_count}" >')
    else:
        lines[-1] = lines[-1] + " >"
    header = "\n".join(lines) + "\n"

    def _fmt(v: float) -> str:
        """Match AFNI's '%g'-ish formatting: integers when exact, else up
        to ~7 sig digits without trailing zeros."""
        if v == int(v) and abs(v) < 1e9:
            return str(int(v))
        return f"{v:.7g}"

    body_lines = []
    for i in range(table.shape[0]):
        row = " " + " ".join(_fmt(table[i, j]) for j in range(table.shape[1]))
        body_lines.append(row)
    body = "\n".join(body_lines) + "\n"

    footer = f"</3dClustSim_NN{nn}>\n"

    out_path.write_text(header + body + footer)


# ---------------------------------------------------------------------------
# Base64 mask blob (matches AFNI's mask_to_b64string)
# ---------------------------------------------------------------------------


def resolve_mask_idcode(mask_path: Path | str | None) -> str:
    """Return an AFNI-style ``AFN_<22 base64 chars>`` IDCODE for the mask.

    Strategy:

    1. If ``3dinfo`` is on PATH and the mask file exists, run
       ``3dinfo -id <file>`` and return its output (real AFNI IDCODE).
    2. Otherwise synthesise a deterministic ID from the absolute mask path
       so reruns produce the same code; AFNI accepts any string here, it
       just uses IDCODE for cross-referencing.
    """
    import hashlib
    import shutil as _shutil
    import subprocess as _subprocess

    if mask_path is None:
        return "AFN_" + hashlib.sha256(b"ffs_perm:no-mask").hexdigest()[:22]

    p = Path(mask_path)
    if _shutil.which("3dinfo") is not None and p.exists():
        try:
            r = _subprocess.run(
                ["3dinfo", "-id", str(p)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            idc = r.stdout.strip()
            if idc and idc.startswith("AFN_"):
                return idc
        except (_subprocess.TimeoutExpired, FileNotFoundError):
            pass

    seed = str(p.resolve() if p.exists() else p).encode()
    digest = hashlib.sha256(seed).digest()[:16]
    return "AFN_" + base64.urlsafe_b64encode(digest).decode().rstrip("=")[:22]


def write_mask_b64(out_path: Path, mask: np.ndarray) -> int:
    """Write the AFNI ``AFNI_CLUSTSIM_MASK`` attribute payload.

    AFNI's ``mask_to_b64string`` (``thd_makemask.c`` / ``zfun.c``) does:

    1. Bit-pack the mask 8 voxels per byte, in **F-order** (AFNI's
       x-fastest internal voxel order).
    2. **zlib**-compress the packed bytes (level 9).
    3. Base64-encode the compressed bytes.
    4. Wrap base64 at **72 chars/line** (LF separators).
    5. Append ``"===<nvox>"`` after the last line so AFNI's loader can
       cross-check the voxel count against the dataset it's attached to.

    Without all five steps AFNI's mask loader silently rejects the
    attribute and the cluster table is ignored — that was the bug behind
    the missing cluster panel.
    """
    flat = mask.astype(bool).ravel(order="F")  # AFNI x-fastest order
    nvox = int(flat.size)
    n_bytes = (nvox + 7) // 8
    # Bit-pack: voxel ii's bit is bit (ii&7) of byte ii>>3.  np.packbits
    # uses big-endian bit order by default; AFNI uses little-endian
    # (1<<(ii&7)), so we set bitorder='little'.
    packed = np.packbits(flat.astype(np.uint8), bitorder="little")
    if packed.size != n_bytes:
        # packbits rounds up; should already match, but be defensive
        packed = np.resize(packed, n_bytes)
    compressed = zlib.compress(packed.tobytes(), 9)
    b64 = base64.b64encode(compressed).decode("ascii")
    wrapped = "\n".join(textwrap.wrap(b64, 72))
    out_path.write_text(wrapped + "===" + str(nvox) + "\n")
    return int(flat.sum())


# ---------------------------------------------------------------------------
# 3drefit dispatch
# ---------------------------------------------------------------------------


def build_refit_commands(
    stat_path: Path,
    niml_files: dict[tuple[int, str], Path],
    mask_b64_path: Path | None,
    brick_labels: list[str] | None = None,
    stat_brick_indices: list[int] | None = None,
    dof: int | None = None,
) -> list[list[str]]:
    """Construct the ``3drefit`` command lines to inject everything.

    AFNI's ``3drefit`` refuses to combine ``-atrstring`` with "modification
    options" (``-relabel_all_str``, ``-substatpar``) in one call, so we
    return a list of independent invocations:

    1. Cluster-sim NIML tables + ``AFNI_CLUSTSIM_MASK`` (atrstrings only).
    2. Sub-brick labels via ``-relabel_all_str`` and per-brick t-stat
       declaration via ``-substatpar <idx> fitt <dof>``.
    """
    cmds: list[list[str]] = []

    cmd1 = ["3drefit"]
    for (nn, sided), path in sorted(niml_files.items()):
        cmd1 += ["-atrstring", f"AFNI_CLUSTSIM_NN{nn}_{sided}", f"file:{path}"]
    if mask_b64_path is not None:
        cmd1 += ["-atrstring", "AFNI_CLUSTSIM_MASK", f"file:{mask_b64_path}"]
    if len(cmd1) > 1:
        cmd1.append(str(stat_path))
        cmds.append(cmd1)

    cmd2 = ["3drefit"]
    if brick_labels:
        for i, lab in enumerate(brick_labels):
            cmd2 += ["-sublabel", str(i), lab]
    if stat_brick_indices and dof is not None:
        for idx in stat_brick_indices:
            cmd2 += ["-substatpar", str(idx), "fitt", str(int(dof))]
    if len(cmd2) > 1:
        cmd2.append(str(stat_path))
        cmds.append(cmd2)

    return cmds


def inject_clustsim_headers(
    stat_path: Path,
    niml_files: dict[tuple[int, str], Path],
    mask_b64_path: Path | None,
    brick_labels: list[str] | None = None,
    stat_brick_indices: list[int] | None = None,
    dof: int | None = None,
) -> None:
    """Inject ClustSim tables, mask, sub-brick labels and t-stat params into the
    stat dataset's AFNI extension — in-script, no ``3drefit``.

    Equivalent to ``3drefit -atrstring AFNI_CLUSTSIM_* file:… -relabel_all_str …
    -substatpar … fitt dof``: the NIML tables and base64 mask are stored as AFNI
    String attributes (``AFNI_suck_file`` reads them verbatim, trailing
    whitespace trimmed), so AFNI re-parses them for cluster thresholding.
    """
    import nibabel as nib

    from fastfuncstuff.io.afni import (
        _set_afni_brick_labels,
        _set_afni_brick_stataux,
        compress_nifti,
        load_nifti,
        replace_afni_extension,
        set_afni_atr,
    )

    path = Path(stat_path)
    # Materialise fully so the save below can overwrite the file without racing
    # a lazy ArrayProxy (same SIGBUS guard as add_fdrcurves_to_nifti).
    src = (
        load_nifti(str(path)) if str(path).endswith(".nii.zst") else nib.load(str(path), mmap=False)
    )
    data = np.asarray(src.dataobj)
    header = src.header.copy()
    affine = src.affine

    # AFNI's `-atrstring … file:` truncates trailing whitespace on the raw file
    # content before storing (AFNI_suck_file); rstrip() matches that.
    for (nn, sided), niml_path in sorted(niml_files.items()):
        content = Path(niml_path).read_text().rstrip()
        set_afni_atr(header, f"AFNI_CLUSTSIM_NN{nn}_{sided}", content, ni_type="String")
    if mask_b64_path is not None:
        set_afni_atr(
            header, "AFNI_CLUSTSIM_MASK", Path(mask_b64_path).read_text().rstrip(), ni_type="String"
        )

    if brick_labels:
        _set_afni_brick_labels(header, brick_labels)
    if stat_brick_indices and dof is not None:
        n_sub = data.shape[3] if data.ndim == 4 else 1
        stataux = {int(idx): (3, (float(int(dof)),)) for idx in stat_brick_indices}  # fitt(dof)
        _set_afni_brick_stataux(header, stataux, n_sub)

    out_img = nib.Nifti1Image(data, affine, header)
    del src

    path_str = str(path)
    if path_str.endswith(".nii.zst"):
        tmp_nii = replace_afni_extension(path_str, ".nii")
        nib.save(out_img, tmp_nii)
        compress_nifti(tmp_nii, path_str, remove_original=True)
    else:
        nib.save(out_img, path_str)


def run_refit(
    stat_path: Path,
    niml_files: dict[tuple[int, str], Path],
    mask_b64_path: Path | None,
    brick_labels: list[str] | None = None,
    stat_brick_indices: list[int] | None = None,
    dof: int | None = None,
    write_script_path: Path | None = None,
    verbose: bool = True,
) -> bool:
    """Inject the ClustSim tables / labels / stat params into ``stat_path``.

    Done in-script (no ``3drefit`` dependency). The equivalent ``3drefit``
    script is still written to ``write_script_path`` as a reproducible record,
    matching AFNI 3dClustSim's ``ppp.3drefit.cmd`` behaviour.
    """
    cmds = build_refit_commands(
        stat_path,
        niml_files,
        mask_b64_path,
        brick_labels=brick_labels,
        stat_brick_indices=stat_brick_indices,
        dof=dof,
    )
    if not cmds:
        return True

    if write_script_path is not None:
        body = "#!/usr/bin/env bash\nset -e\n"
        for c in cmds:
            body += " ".join(_quote(a) for a in c) + "\n"
        write_script_path.write_text(body)
        write_script_path.chmod(0o755)

    try:
        inject_clustsim_headers(
            stat_path,
            niml_files,
            mask_b64_path,
            brick_labels=brick_labels,
            stat_brick_indices=stat_brick_indices,
            dof=dof,
        )
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"In-script ClustSim header injection failed: {e}", file=sys.stderr)
            if write_script_path is not None:
                print(
                    f"  Run the fallback script when able:\n    {write_script_path}",
                    file=sys.stderr,
                )
        return False
    return True


def _quote(s: str) -> str:
    if any(c in s for c in " \t\"'$`\\"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s
