"""ffs_info — dataset header reporting, a drop-in replacement for ``3dinfo``.

Two modes:

* **Value flags** (``-nv``, ``-tr``, ``-ad3``, …) print bare values, one line per
  dataset, tab-separated in the order the flags were given. This is the
  scripting contract ``3dinfo`` established and the generated pipelines depend
  on; matching it exactly is the point.
* **No flags** prints an organised human report.

Everything except ``-dmin``/``-dmax``/``-vis`` is answered from the header alone,
which for a ``.nii.zst`` means a few KB off the front of the file rather than a
full decompress.
"""

from __future__ import annotations

import argparse
import os
import sys

from fastfuncstuff.io.dsetinfo import (
    _XFORM_NAMES,
    DatasetInfo,
    data_range,
    read_info,
    read_volume,
    same_grid,
)

_ORIENT_WORDS = {
    "R": "Right", "L": "Left", "A": "Anterior", "P": "Posterior", "I": "Inferior", "S": "Superior",
}  # fmt: skip


# ---------------------------------------------------------------- value flags


def _fmt_f(x: float) -> str:
    return f"{x:.6f}"


def _join(vals, sep: str = "\t") -> str:
    return sep.join(_fmt_f(v) if isinstance(v, float) else str(v) for v in vals)


# flag → (callable(info, args) -> str, help text). Order here is the -help order.
_VALUE_FLAGS: dict[str, tuple] = {
    "-exists": (lambda i, a: int(i.exists), "1 if the dataset can be read, else 0"),
    "-iname": (lambda i, a: i.iname, "input name as given on the command line"),
    "-prefix": (lambda i, a: i.prefix, "file name without directories"),
    "-smode": (lambda i, a: i.storage, "storage mode (NIFTI / BRIK)"),
    # AFNI reports the datum per sub-brick; ours is uniform, but the shape of the
    # answer has to match or a caller splitting on the delimiter gets a surprise.
    "-datum": (lambda i, a: a.sb_delim.join([i.datum] * max(i.shape[3], 1)), "voxel data type"),
    "-is_nifti": (lambda i, a: int(i.is_nifti), "1 if stored as NIfTI"),
    "-ni": (lambda i, a: i.shape[0], "number of voxels along the i axis"),
    "-nj": (lambda i, a: i.shape[1], "number of voxels along the j axis"),
    "-nk": (lambda i, a: i.shape[2], "number of voxels along the k axis"),
    "-nv": (lambda i, a: i.shape[3], "number of sub-bricks / volumes"),
    "-nt": (lambda i, a: i.shape[3], "number of time points (same as -nv)"),
    "-n4": (lambda i, a: _join(i.shape), "ni nj nk nv"),
    "-nvi": (lambda i, a: i.shape[3] - 1, "number of sub-bricks minus 1"),
    "-nti": (lambda i, a: i.shape[3] - 1, "number of time points minus 1"),
    "-nijk": (lambda i, a: i.shape[0] * i.shape[1] * i.shape[2], "voxels per volume"),
    "-adi": (lambda i, a: i.zooms[0], "voxel size along i (absolute)"),
    "-adj": (lambda i, a: i.zooms[1], "voxel size along j (absolute)"),
    "-adk": (lambda i, a: i.zooms[2], "voxel size along k (absolute)"),
    "-ad3": (lambda i, a: _join(i.zooms), "voxel sizes (absolute)"),
    "-di": (lambda i, a: i.signed_steps[0], "voxel step along i (signed)"),
    "-dj": (lambda i, a: i.signed_steps[1], "voxel step along j (signed)"),
    "-dk": (lambda i, a: i.signed_steps[2], "voxel step along k (signed)"),
    "-d3": (lambda i, a: _join(i.signed_steps), "voxel steps (signed)"),
    "-oi": (lambda i, a: i.origin[0], "origin (AFNI x) of voxel (0,0,0)"),
    "-oj": (lambda i, a: i.origin[1], "origin (AFNI y) of voxel (0,0,0)"),
    "-ok": (lambda i, a: i.origin[2], "origin (AFNI z) of voxel (0,0,0)"),
    "-o3": (lambda i, a: _join(i.origin), "origin of voxel (0,0,0)"),
    "-extent": (lambda i, a: _join(i.extent), "spatial extent: R L A P I S"),
    "-fov": (lambda i, a: _join(i.fov), "field of view in mm (nvox * voxel size)"),
    "-tr": (lambda i, a: i.tr, "TR in seconds (0 if not a time series)"),
    "-duration": (lambda i, a: i.duration, "TR * number of volumes, in seconds"),
    "-orient": (lambda i, a: i.orient, "3-letter orientation code, AFNI convention"),
    "-space": (lambda i, a: i.space, "template space"),
    "-obliquity": (lambda i, a: f"{i.obliquity:.3f}", "degrees from plumb"),
    "-is_oblique": (lambda i, a: int(i.is_oblique), "1 if the dataset is oblique"),
    "-slice_timing": (
        lambda i, a: _join(i.slice_timing or [], a.sb_delim),
        "per-slice acquisition offsets",
    ),
    # An unlabelled sub-brick prints "?" (AFNI's placeholder), one per volume.
    "-label": (
        lambda i, a: a.sb_delim.join(i.labels or ["?"] * max(i.shape[3], 1)),
        "sub-brick labels",
    ),
    "-history": (lambda i, a: i.history, "processing history"),
    "-dmin": (lambda i, a: data_range(i.iname)[0], "minimum voxel value (reads the data)"),
    "-dmax": (lambda i, a: data_range(i.iname)[1], "maximum voxel value (reads the data)"),
}


def _requested_flags(argv: list[str]) -> list[str]:
    """Value flags in command-line order, as ``3dinfo`` reports them.

    argparse throws the ordering away, and the order is part of the contract for
    anything doing ``read a b c <<< $(ffs_info -ni -nj -nk dset)``.
    """
    seen: list[str] = []
    for tok in argv:
        if not tok.startswith("-"):
            continue
        norm = "-" + tok[1:].replace("-", "_")  # accept -slice-timing and -slice_timing
        if norm in _VALUE_FLAGS and norm not in seen:
            seen.append(norm)
    return seen


# ------------------------------------------------------------- pretty report


class _Style:
    """ANSI styling that evaporates when stdout is not a terminal."""

    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def bold(self, s: str) -> str:
        return self._w("1", s)

    def dim(self, s: str) -> str:
        return self._w("2", s)

    def key(self, s: str) -> str:
        return self._w("36", s)

    def warn(self, s: str) -> str:
        return self._w("33", s)


def _human_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{int(n)} B"


def _human_time(sec: float) -> str:
    if sec < 90:
        return f"{sec:.1f} s"
    return f"{sec / 60:.1f} min"


def format_report(info: DatasetInfo, st: _Style, show_history: bool = True) -> str:
    """The no-flags human view: one aligned block per dataset."""
    rows: list[tuple[str, str]] = []

    stored = [info.storage]
    if info.compression:
        stored.append(info.compression)
    raw = info.shape[0] * info.shape[1] * info.shape[2] * info.shape[3] * info.itemsize
    size = f"{_human_bytes(info.file_bytes)} on disk"
    if info.compression and raw:
        size += f" · {_human_bytes(raw)} raw ({raw / max(info.file_bytes, 1):.1f}× )".replace(
            " )", ")"
        )
    rows.append(("storage", f"{' · '.join(stored)} · {info.datum} · {size}"))

    nx, ny, nz, nv = info.shape
    grid = f"{nx} × {ny} × {nz}"
    grid += f" × {nv} volumes" if nv > 1 else "  (single volume)"
    if info.selector is not None:
        grid += st.dim("   [selector applied]")
    rows.append(("grid", grid))

    dx, dy, dz = info.zooms
    fx, fy, fz = info.fov
    rows.append(
        (
            "voxels",
            f"{dx:.3f} × {dy:.3f} × {dz:.3f} mm"
            + st.dim(f"    fov {fx:.1f} × {fy:.1f} × {fz:.1f} mm"),
        )
    )

    if info.tr > 0:
        time_bits = [f"TR {info.tr:.4g} s", f"{nv} volumes", _human_time(info.duration)]
        if info.slice_timing:
            t = info.slice_timing
            pattern = f", {info.slice_order}" if info.slice_order else ""
            time_bits.append(f"slice timing: {len(t)} slices {min(t):.3f}–{max(t):.3f} s{pattern}")
        rows.append(("time", " · ".join(time_bits)))

    axes = " ".join(
        f"{ax}: {_ORIENT_WORDS[c]}→{_ORIENT_WORDS[flip]}"
        for ax, c, flip in zip("ijk", info.orient, _flip(info.orient), strict=True)
    )
    rows.append(("orient", f"{st.bold(info.orient)}   " + st.dim(axes)))

    # The s/qform codes only mean something for a NIfTI; a BRIK has neither.
    xform = (
        st.dim(
            f"sform={info.sform_code} ({_XFORM_NAMES.get(info.sform_code, '?')})"
            f"  qform={info.qform_code} ({_XFORM_NAMES.get(info.qform_code, '?')})"
        )
        if info.is_nifti
        else ""
    )
    rows.append(("space", f"{info.space}   {xform}".rstrip()))

    tilt = (
        st.warn(f"oblique — {info.obliquity:.3f}° from plumb")
        if info.is_oblique
        else f"plumb ({info.obliquity:.3f}°)"
    )
    rows.append(("tilt", tilt))

    ox, oy, oz = info.origin
    rows.append(("origin", f"ijk(0,0,0) → x {ox:.3f}  y {oy:.3f}  z {oz:.3f}  " + st.dim("[AFNI]")))

    e = info.extent
    rows.append(
        (
            "extent",
            f"R {e[0]:8.3f} → L {e[1]:8.3f}   A {e[2]:8.3f} → P {e[3]:8.3f}"
            f"   I {e[4]:8.3f} → S {e[5]:8.3f}",
        )
    )

    if info.labels and (len(info.labels) > 1 or info.labels[0] != "?"):
        shown = ", ".join(info.labels[:8])
        if len(info.labels) > 8:
            shown += st.dim(f" … (+{len(info.labels) - 8})")
        rows.append(("labels", shown))

    if info.scl_slope != 1.0 or info.scl_inter != 0.0:
        rows.append(("scaling", f"slope {info.scl_slope:g}  inter {info.scl_inter:g}"))

    if info.descrip:
        rows.append(("descrip", info.descrip))

    width = max(len(k) for k, _ in rows)
    body = "\n".join(f"  {st.key(k.ljust(width))}  {v}" for k, v in rows)

    header = st.bold(str(info.iname))
    rule = st.dim("─" * min(len(str(info.iname)) + 2, 78))
    out = f"{header}\n{rule}\n{body}"

    if show_history and info.history:
        lines = info.history.strip().splitlines()
        tail = lines[-4:]
        out += "\n" + st.dim("  history") + "\n"
        out += "\n".join("    " + st.dim(line.strip()) for line in tail)
        if len(lines) > len(tail):
            out += st.dim(f"\n    ({len(lines) - len(tail)} earlier entries)")
    return out


def _flip(orient: str) -> str:
    flip = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    return "".join(flip[c] for c in orient)


# --------------------------------------------------------------------- -vis


def _visualize(info: DatasetInfo, args) -> str:
    """Draw one volume — read on its own, not by loading the whole series."""
    from fastfuncstuff.termvis import orthoview, supports_truecolor, to_ras

    data, _ = read_volume(info.iname, args.vis_vol)
    data, zooms = to_ras(data, info.affine)
    color = not args.vis_ascii and supports_truecolor()
    return orthoview(
        data,
        zooms,
        slices=tuple(args.vis_slice) if args.vis_slice else None,
        width=args.vis_width,
        color=color,
        clip=tuple(args.vis_clip),
    )


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_info",
        description=(
            "Report dataset header information (a 3dinfo replacement that reads "
            ".nii, .nii.gz, .nii.zst and AFNI HEAD/BRIK without decompressing the "
            "image data), optionally drawing the volume in the terminal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ffs_info run.nii.zst                  # full report\n"
            "  ffs_info -nv run.nii.zst              # just the volume count\n"
            "  ffs_info -ni -nj -nk -tr run.nii.zst  # tab-separated, in this order\n"
            "  ffs_info -vis anat.nii.gz             # 3-plane greyscale view\n"
        ),
    )
    p.add_argument("dsets", nargs="*", help="one or more datasets ([selector] suffixes honoured)")

    values = p.add_argument_group("value flags (bare output, one line per dataset)")
    for flag, (_, helptext) in _VALUE_FLAGS.items():
        names = [flag] + ([flag.replace("_", "-")] if "_" in flag else [])
        values.add_argument(*names, action="store_true", help=helptext)

    multi = p.add_argument_group("multi-dataset")
    multi.add_argument(
        "-same_grid",
        "-same-grid",
        action="store_true",
        help="1 if all datasets share one voxel grid (shape and affine), else 0",
    )

    fmt = p.add_argument_group("formatting")
    fmt.add_argument(
        "-sb_delim", "-sb-delim", default="|", help="delimiter for per-sub-brick lists"
    )
    fmt.add_argument(
        "-header_line", "-header-line", action="store_true", help="print a column-name header line"
    )
    fmt.add_argument("-no_hist", "-no-hist", action="store_true", help="omit history in the report")
    fmt.add_argument("-no_color", "-no-color", action="store_true", help="never emit ANSI colour")

    vis = p.add_argument_group("terminal visualization")
    vis.add_argument(
        "-vis", action="store_true", help="draw axial/coronal/sagittal greyscale views"
    )
    vis.add_argument("-vis_vol", "-vis-vol", type=int, default=0, help="volume index to draw")
    vis.add_argument(
        "-vis_slice",
        "-vis-slice",
        type=int,
        nargs=3,
        metavar=("I", "J", "K"),
        help="RAS-ordered slice indices to draw (default: the middle of each axis)",
    )
    vis.add_argument("-vis_width", "-vis-width", type=int, help="total width in characters")
    vis.add_argument(
        "-vis_ascii", "-vis-ascii", action="store_true", help="ASCII ramp instead of colour blocks"
    )
    vis.add_argument(
        "-vis_clip",
        "-vis-clip",
        type=float,
        nargs=2,
        default=[1.0, 99.0],
        metavar=("LO", "HI"),
        help="intensity percentiles mapped to black/white (default 1 99)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    if not args.dsets:
        build_parser().print_help()
        return 1

    flags = _requested_flags(argv)
    color = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    st = _Style(color)

    infos: list[DatasetInfo] = []
    status = 0
    for dset in args.dsets:
        info = read_info(dset)
        infos.append(info)
        if not info.exists and not flags and not args.same_grid:
            print(f"** cannot read: {dset}", file=sys.stderr)
            status = 1

    if args.same_grid:
        print(int(all(i.exists for i in infos) and same_grid(infos)))
        return status

    if flags:
        if args.header_line:
            print("\t".join(f.lstrip("-") for f in flags))
        for info in infos:
            if not info.exists:
                # 3dinfo prints NO-DSET rather than dying, so a loop over a glob
                # with one bad entry still produces one line per input.
                print("\t".join(["NO-DSET"] * len(flags)))
                status = 1
                continue
            vals = [_VALUE_FLAGS[f][0](info, args) for f in flags]
            print("\t".join(_fmt_f(v) if isinstance(v, float) else str(v) for v in vals))
        return status

    for n, info in enumerate(infos):
        if not info.exists:
            continue
        if n:
            print()
        print(format_report(info, st, show_history=not args.no_hist))
        if args.vis:
            print()
            print(_visualize(info, args))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
