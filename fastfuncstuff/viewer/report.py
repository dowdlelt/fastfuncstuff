"""A written description of what a viewer session currently is.

For handing to someone who cannot see the screen. A screenshot shows what is
drawn; the questions that actually matter when something looks wrong are which
grid each layer is on, which one is defining the display, what the mode is
reading and what has been done to get here -- none of which a picture answers.

Two halves, and the second one is the valuable one:

* **The state**, laid out flat: grid, crosshair, every layer with its shape,
  affine, voxel size and how it is drawn, and what the mode is doing.
* **The script**, which is the whole recorded command history and so is a
  reproduction rather than a description. Replayed against the same files it
  rebuilds the session exactly.

No Qt here, so it can be written from a test, a headless session or a button.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastfuncstuff.viewer.layers import Layer
    from fastfuncstuff.viewer.session import ViewerSession


def _affine_lines(affine: np.ndarray, indent: str = "      ") -> list[str]:
    a = np.asarray(affine, dtype=float)
    return [indent + "  ".join(f"{v:9.4f}" for v in row) for row in a[:3]]


def _zooms(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel size along each index axis, obliquity included.

    The column norm rather than the diagonal: an oblique dataset has its size
    spread across the row, and reading the diagonal reports a 3 mm voxel as
    2.6 mm without saying why.
    """
    a = np.asarray(affine, dtype=float)[:3, :3]
    return (
        float(np.linalg.norm(a[:, 0])),
        float(np.linalg.norm(a[:, 1])),
        float(np.linalg.norm(a[:, 2])),
    )


def _obliquity(affine: np.ndarray) -> float:
    """Degrees between the layer's axes and the scanner's, 0 for a cardinal grid."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    cols = a / np.linalg.norm(a, axis=0, keepdims=True).clip(min=1e-12)
    # The best each axis manages to align with any cardinal direction; the
    # worst of those three is what "how oblique is it" means.
    best = np.abs(cols).max(axis=0).clip(max=1.0)
    return float(np.degrees(np.arccos(best.min())))


def _layer_block(session: ViewerSession, layer: Layer, *, index: int, is_grid: bool) -> list[str]:
    zx, zy, zz = _zooms(layer.affine)
    marks = []
    if is_grid:
        marks.append("DEFINES DISPLAY GRID")
    if not layer.visible:
        marks.append("hidden")
    if layer.roi:
        marks.append("roi")
    if layer.time_linked:
        marks.append("time-linked")
    out = [
        f"  [{index}] {layer.key}  {layer.name}" + (f"   <{', '.join(marks)}>" if marks else ""),
        f"      path      {layer.path}",
        f"      source    {layer.source}",
        f"      shape     {layer.shape}  x {layer.n_volumes} volume(s)",
        f"      voxel     {zx:.4f} x {zy:.4f} x {zz:.4f} mm"
        + (
            f"   oblique {_obliquity(layer.affine):.2f} deg"
            if _obliquity(layer.affine) > 0.01
            else ""
        ),
        "      affine",
        *_affine_lines(layer.affine),
    ]
    resample = session.resample_mode(layer)
    out.append(
        f"      drawn     {resample} resample, colormap {layer.colormap}, "
        f"opacity {layer.opacity:.2f}, sign {layer.sign_mode}"
    )
    lo = "auto" if layer.range_lo is None else f"{layer.range_lo:.6g}"
    hi = "auto" if layer.range_hi is None else f"{layer.range_hi:.6g}"
    out.append(
        f"      range     {lo} .. {hi}"
        + ("  mirrored" if layer.range_mirror else "")
        + f"   threshold {layer.threshold:.6g} (alpha {layer.alpha_mode})"
    )
    out.append(
        f"      sub-brick {layer.sub_brick()}"
        f"   threshold on {layer.sub_brick(layer.threshold_brick)}"
        f" (follow {layer.threshold_follow})"
    )
    if layer.labels:
        shown = ", ".join(layer.labels[:8]) + (" ..." if len(layer.labels) > 8 else "")
        out.append(f"      labels    {shown}")
    if layer.stataux:
        out.append(f"      stataux   {dict(sorted(layer.stataux.items()))}")
    return out


def describe(session: ViewerSession, *, script: bool = True) -> str:
    """Everything about this session, as text.

    ``script=False`` leaves out the recorded history, which is the only part
    that can be long -- a review session is thousands of crosshair moves.
    """
    st = session.state
    out: list[str] = ["# ffs viewer session report", ""]

    out.append("## session")
    out.append(f"  controller   {session.label or '(unlabelled)'}")
    out.append(f"  device       compute {session.store.device}, display {session.display_device}")
    out.append(f"  directory    {session.catalog_dir or '(none)'}")
    out.append(f"  catalog      {len(session.catalog)} dataset(s)")
    out.append("")

    out.append("## display grid")
    if st.grid is None:
        out.append("  (nothing loaded)")
    else:
        base = st.layers.base
        zx, zy, zz = _zooms(st.grid.affine)
        out.append(f"  from         {base.name if base else '(none)'}  [bottom of the stack]")
        out.append(f"  shape        {st.grid.shape}")
        out.append(f"  voxel        {zx:.4f} x {zy:.4f} x {zz:.4f} mm")
        out.append("  affine")
        out.extend(_affine_lines(st.grid.affine, indent="    "))
        mm = st.crosshair_mm
        out.append(
            f"  crosshair    ijk {st.crosshair}"
            + (f"   mm ({mm[0]:.2f}, {mm[1]:.2f}, {mm[2]:.2f})" if mm else "")
        )
        if st.seed is not None:
            out.append(f"  seed         ijk {st.seed}")
        out.append(f"  time index   {st.time_index} / {st.max_time_index()}")
    out.append("")

    out.append(f"## layers  ({len(st.layers)}, bottom first)")
    if not len(st.layers):
        out.append("  (none)")
    base_key = st.layers.base.key if st.layers.base is not None else None
    for i, layer in enumerate(st.layers):
        out.extend(_layer_block(session, layer, index=i, is_grid=layer.key == base_key))
        out.append("")
    out.append(f"  selected     {st.selected or '(none)'}")
    out.append("")

    out.append("## mode")
    mode = session.mode
    out.append(f"  active       {mode.name}   (reads: {mode.input_kind})")
    resolved = session.input_layer()
    out.append(
        f"  input        {st.input_key or '(auto)'} -> {resolved.name if resolved else 'none'}"
    )
    out.append(f"  candidates   {[ly.name for ly in session.input_candidates()] or '(none)'}")
    if mode.params:
        out.append("  parameters")
        for name in sorted(mode.params):
            out.append(f"      {name:<16} {mode.params[name]!r}")
    if mode.status():
        out.append(f"  status       {mode.status()}")
    out.append("")

    out.append(f"## windows  ({len(list(st.viewports))})")
    for vp in st.viewports:
        out.append(f"  {vp.id:<8} {vp.kind:<10} {vp.plane}")
    if not len(list(st.viewports)):
        out.append("  (none)")
    out.append("")

    if script:
        out.append("## script  (replays this session against the same files)")
        out.append("")
        out.append(session.to_script(header=None).rstrip())
        out.append("")

    return "\n".join(out)
