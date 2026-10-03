"""FreeSurfer surfaces: read them in scanner space, write edited copies faithfully.

Every surface of a hemisphere -- white, pial, smoothwm, inflated, sphere -- is
the same mesh with the same vertex numbering and only the positions changed.
That is what lets a value sampled between white and pial be shown on the
inflated surface, and it is also why an edit that moves vertices without
touching the faces leaves every derived surface valid.

Coordinates in a FreeSurfer surface file are *tkregister* RAS: the conformed
volume's frame with its centre at the origin. The volume geometry stored in
the file's trailer says where that volume sat in the scanner, so
:func:`tkr_to_scanner` turns it into the scanner RAS that every NIfTI affine
in this toolbox (and the viewer's display grid) already uses.

Writing never round-trips through a parser: :func:`write_surface_like` copies
the original file byte for byte and replaces only the coordinate block, so the
volume geometry, the command-line tags and anything a later FreeSurfer
version adds survive an edit unchanged.

Torch-free on purpose -- this is header-speed I/O and the viewer's load path
must not pay for a torch import to read a mesh.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TRIANGLE_MAGIC = b"\xff\xff\xfe"
_PATCH_MAGIC = -1

#: Surfaces read for each hemisphere when present. ``white`` and ``pial`` are
#: where data is sampled; the rest are shapes to display it on.
DEFAULT_STATES = ("white", "pial", "smoothwm", "inflated", "sphere")


@dataclass
class SurfaceFile:
    """One surface file: positions in the file's own tkregister RAS."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    volume_info: dict[str, object]
    #: Byte offset of the coordinate block, so a writer can splice new
    #: positions into an otherwise untouched copy.
    coord_offset: int


def _triangle_layout(raw: bytes) -> tuple[int, int, int]:
    """(n_vertices, n_faces, coord_offset) of a triangle-format surface."""
    if raw[:3] != _TRIANGLE_MAGIC:
        raise ValueError("not a FreeSurfer triangle surface (bad magic number)")
    # Created-by line terminated by "\n\n"; FreeSurfer's own reader skips
    # exactly two newlines, and so must we -- a comment can be empty.
    end = raw.index(b"\n", 3)
    end = raw.index(b"\n", end + 1)
    nv, nf = np.frombuffer(raw, ">i4", count=2, offset=end + 1)
    return int(nv), int(nf), end + 9


def read_surface(path: str | os.PathLike) -> SurfaceFile:
    """Read a triangle surface, keeping enough layout to rewrite it in place."""
    import nibabel.freesurfer as nfs

    raw = Path(path).read_bytes()
    nv, nf, offset = _triangle_layout(raw)
    vertices = np.frombuffer(raw, ">f4", count=nv * 3, offset=offset).reshape(nv, 3)
    faces = np.frombuffer(raw, ">i4", count=nf * 3, offset=offset + nv * 12).reshape(nf, 3)
    # nibabel already parses the trailer's volume-geometry tags; reuse it.
    _, _, meta = nfs.read_geometry(str(path), read_metadata=True)
    return SurfaceFile(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        volume_info=dict(meta),
        coord_offset=offset,
    )


def write_surface_like(
    path: str | os.PathLike, template: str | os.PathLike, vertices: np.ndarray
) -> None:
    """Write ``template`` to ``path`` with only its vertex positions replaced.

    ``vertices`` are tkregister RAS, the file's own frame. Refuses to write
    over the template itself: an edited surface is always a copy.
    """
    path, template = Path(path), Path(template)
    if path.exists() and path.resolve() == template.resolve():
        raise ValueError(f"refusing to overwrite the original surface {template}")
    raw = bytearray(template.read_bytes())
    nv, _, offset = _triangle_layout(bytes(raw))
    vertices = np.asarray(vertices)
    if vertices.shape != (nv, 3):
        raise ValueError(f"expected ({nv}, 3) vertices to match {template}, got {vertices.shape}")
    raw[offset : offset + nv * 12] = vertices.astype(">f4").tobytes()
    path.write_bytes(bytes(raw))


def read_patch(
    path: str | os.PathLike, n_vertices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ``?h.*.patch.3d`` flat patch.

    Returns ``(coords, in_patch, border)``: ``coords`` is (V, 3) over the
    *whole* hemisphere with zeros outside the patch, so it lines up with
    every other surface by vertex index; ``in_patch`` and ``border`` are
    boolean masks. Vertex numbers are stored one-based and negated on the
    cut's border.
    """
    raw = Path(path).read_bytes()
    magic, count = np.frombuffer(raw, ">i4", count=2)
    if magic != _PATCH_MAGIC:
        raise ValueError(f"{path}: not a FreeSurfer patch file")
    rec = np.frombuffer(
        raw, dtype=[("v", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")], count=count, offset=8
    )
    ids = np.abs(rec["v"].astype(np.int64)) - 1
    if ids.size and (ids.min() < 0 or ids.max() >= n_vertices):
        raise ValueError(f"{path}: vertex ids outside the {n_vertices}-vertex surface")
    coords = np.zeros((n_vertices, 3), np.float32)
    coords[ids, 0], coords[ids, 1], coords[ids, 2] = rec["x"], rec["y"], rec["z"]
    in_patch = np.zeros(n_vertices, bool)
    in_patch[ids] = True
    border = np.zeros(n_vertices, bool)
    border[ids[rec["v"] < 0]] = True
    return coords, in_patch, border


def _volume_geometry(info: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """(vox2ras, vox2ras_tkr) of the volume a surface was made from."""
    try:
        dims = np.asarray(info["volume"], float)
        vs = np.asarray(info["voxelsize"], float)
        cosines = np.column_stack([info["xras"], info["yras"], info["zras"]]).astype(float)
        cras = np.asarray(info["cras"], float)
    except KeyError as missing:
        raise ValueError(
            f"surface has no volume geometry ({missing} missing); cannot place it in scanner space"
        ) from None
    centre = dims / 2.0
    vox2ras = np.eye(4)
    vox2ras[:3, :3] = cosines * vs
    vox2ras[:3, 3] = cras - vox2ras[:3, :3] @ centre
    # tkregister space: the conformed (LIA) axes with no rotation and the
    # volume centre at the origin, whatever the scanner did.
    tkr = np.eye(4)
    tkr[:3, :3] = [[-vs[0], 0, 0], [0, 0, vs[2]], [0, -vs[1], 0]]
    tkr[:3, 3] = -tkr[:3, :3] @ centre
    return vox2ras, tkr


def tkr_to_scanner(info: dict[str, object]) -> np.ndarray:
    """4x4 taking a surface file's coordinates to scanner RAS."""
    vox2ras, tkr = _volume_geometry(info)
    return vox2ras @ np.linalg.inv(tkr)


def _apply(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (xyz @ affine[:3, :3].T.astype(np.float32) + affine[:3, 3].astype(np.float32)).astype(
        np.float32
    )


@dataclass
class FlatPatch:
    name: str
    coords: np.ndarray  # (V, 3) over the whole hemisphere, zero outside
    in_patch: np.ndarray  # (V,) bool
    border: np.ndarray  # (V,) bool

    def faces(self, faces: np.ndarray) -> np.ndarray:
        """The faces whose three corners all lie in the patch."""
        return faces[self.in_patch[faces].all(axis=1)]


@dataclass
class Hemisphere:
    """One hemisphere: a single mesh and the positions it takes in each state.

    ``states`` positions are scanner RAS for the anatomical surfaces (white,
    pial, smoothwm) and the file's own frame for the display-only shapes
    (inflated, sphere), which have no meaning in the scanner -- they are
    shifted to scanner RAS too, so a hemisphere sits roughly where its
    anatomy does, but nothing should sample a volume at them.
    """

    name: str
    faces: np.ndarray
    states: dict[str, np.ndarray]
    tkr_to_scanner: np.ndarray
    #: Path each state was read from; the template an edited copy is spliced
    #: into.
    paths: dict[str, Path] = field(default_factory=dict)
    morph: dict[str, np.ndarray] = field(default_factory=dict)
    patches: dict[str, FlatPatch] = field(default_factory=dict)
    #: ``label/?h.cortex.label`` as a vertex mask, or ``None`` when absent.
    #: The medial wall (callosum, the brainstem cut) is mesh but not cortex:
    #: white and pial coincide there, and anything sampled between them is
    #: meaningless -- QC and depth sampling should skip it.
    cortex: np.ndarray | None = None

    @property
    def n_vertices(self) -> int:
        return self.states["white"].shape[0]

    def save_state(self, state: str, path: str | os.PathLike) -> None:
        """Write ``states[state]`` as a copy of the file it was read from.

        The edit is applied as a *displacement* onto the template's own
        coordinates rather than by mapping scanner positions back: the
        float32 round trip through scanner space perturbs every vertex in
        the last bit, and a vertex nobody touched should come out
        bit-identical.
        """
        self.save_positions(state, path, self.states[state])

    def original(self, state: str) -> np.ndarray:
        """``state`` as its file has it (scanner RAS), whatever has been edited since."""
        return _apply(self.tkr_to_scanner, read_surface(self.paths[state]).vertices)

    def save_positions(self, state: str, path: str | os.PathLike, positions: np.ndarray) -> None:
        """Write ``positions`` (scanner RAS) as a copy of ``state``'s file; see :meth:`save_state`."""
        template = self.paths[state]
        tkr = read_surface(template).vertices
        delta = (np.asarray(positions) - _apply(self.tkr_to_scanner, tkr)).astype(np.float64)
        moved = np.any(delta != 0, axis=1)
        rotation = np.linalg.inv(self.tkr_to_scanner)[:3, :3]
        out = tkr.copy()
        out[moved] = (tkr[moved] + delta[moved] @ rotation.T).astype(np.float32)
        write_surface_like(path, template, out)


def load_hemisphere(
    subject_dir: str | os.PathLike,
    hemi: str,
    states: tuple[str, ...] = DEFAULT_STATES,
    morph: tuple[str, ...] = ("curv", "sulc", "thickness"),
    patches: bool = True,
) -> Hemisphere:
    """Read one hemisphere of a FreeSurfer subject into scanner RAS.

    ``white`` is required -- it is the reference every other surface is
    checked against. Missing optional states and overlays are skipped; a
    state whose vertex count or faces disagree with ``white`` is an error,
    because sampling by vertex index would then be silently wrong.
    """
    surf = Path(subject_dir) / "surf"
    ref_path = surf / f"{hemi}.white"
    ref = read_surface(ref_path)
    to_scanner = tkr_to_scanner(ref.volume_info)
    nv = ref.vertices.shape[0]
    hemisphere = Hemisphere(
        name=hemi,
        faces=ref.faces,
        states={"white": _apply(to_scanner, ref.vertices)},
        tkr_to_scanner=to_scanner,
        paths={"white": ref_path},
    )
    for state in states:
        if state == "white":
            continue
        path = surf / f"{hemi}.{state}"
        if not path.exists():
            continue
        s = read_surface(path)
        if s.vertices.shape[0] != nv or not np.array_equal(s.faces, ref.faces):
            raise ValueError(f"{path} is not the same mesh as {ref_path}")
        hemisphere.states[state] = _apply(to_scanner, s.vertices)
        hemisphere.paths[state] = path
    if "pial" not in hemisphere.states:
        raise FileNotFoundError(f"{surf / f'{hemi}.pial'} is required alongside the white surface")

    import nibabel.freesurfer as nfs

    for name in morph:
        path = surf / f"{hemi}.{name}"
        if path.exists():
            values = nfs.read_morph_data(str(path)).astype(np.float32)
            if values.shape[0] == nv:
                hemisphere.morph[name] = values
    label = Path(subject_dir) / "label" / f"{hemi}.cortex.label"
    if label.exists():
        ids = nfs.read_label(str(label))
        mask = np.zeros(nv, bool)
        mask[ids[(ids >= 0) & (ids < nv)]] = True
        hemisphere.cortex = mask
    if patches:
        for path in sorted(surf.glob(f"{hemi}.*.patch.3d")):
            name = path.name[len(hemi) + 1 : -len(".patch.3d")]
            coords, in_patch, border = read_patch(path, nv)
            hemisphere.patches[name] = FlatPatch(name, coords, in_patch, border)
    return hemisphere


#: Annotation labels that name no region: drawn transparent, never reported.
NOT_A_REGION = frozenset({"unknown", "Unknown", "Medial_wall", "corpuscallosum"})


@dataclass
class Annotation:
    """A surface parcellation: one label index per vertex, and the label table."""

    labels: np.ndarray  # (V,) int, index into names/rgba; -1 = unlabelled
    names: list[str]
    rgba: np.ndarray  # (K, 4) uint8

    def name_at(self, vertex: int) -> str | None:
        k = int(self.labels[int(vertex)])
        if k < 0 or k >= len(self.names) or self.names[k] in NOT_A_REGION:
            return None
        return self.names[k]

    def vertex_rgba(self) -> np.ndarray:
        """``(V, 4)`` uint8 colours; vertices in no region are transparent."""
        out = np.zeros((self.labels.size, 4), np.uint8)
        real = np.array([n not in NOT_A_REGION for n in self.names] + [False])
        idx = np.where((self.labels >= 0) & (self.labels < len(self.names)), self.labels, -1)
        ok = real[idx]
        out[ok] = self.rgba[idx[ok]]
        return out


def read_annotation(path: str | os.PathLike, n_vertices: int | None = None) -> Annotation:
    """Read ``?h.<name>.annot``. Vertices it does not cover are unlabelled."""
    import nibabel.freesurfer as nfs

    labels, ctab, names = nfs.read_annot(str(path))
    labels = np.asarray(labels, np.int64)
    if n_vertices is not None and labels.size != n_vertices:
        raise ValueError(f"{path} labels {labels.size} vertices, surface has {n_vertices}")
    rgba = np.zeros((ctab.shape[0], 4), np.uint8)
    rgba[:, :3] = np.clip(ctab[:, :3], 0, 255)
    rgba[:, 3] = 255
    return Annotation(labels, [n.decode() if isinstance(n, bytes) else str(n) for n in names], rgba)


def available_annotations(subject_dir: str | os.PathLike) -> list[str]:
    """Parcellations present for both hemispheres, e.g. ``aparc``, ``aparc.a2009s``."""
    label = Path(subject_dir) / "label"
    lh = {p.name[3:-6] for p in label.glob("lh.*.annot")}
    rh = {p.name[3:-6] for p in label.glob("rh.*.annot")}
    order = ["aparc", "aparc.a2009s", "aparc.DKTatlas"]
    both = lh & rh
    return [n for n in order if n in both] + sorted(both - set(order))


def available_volume_atlases(subject_dir: str | os.PathLike) -> list[str]:
    """Label volumes under ``mri/``, e.g. ``aparc+aseg``, ``aseg``."""
    mri = Path(subject_dir) / "mri"
    names = {
        p.name[: -len(".mgz")] for p in mri.glob("*.mgz") if "aseg" in p.name or "aparc" in p.name
    }
    names -= {n for n in names if n.startswith("wm.") or "presurf" in n or n.endswith(".auto")}
    order = ["aparc+aseg", "aparc.a2009s+aseg", "aparc.DKTatlas+aseg", "aseg"]
    return [n for n in order if n in names] + sorted(names - set(order))


def read_color_lut(
    path: str | os.PathLike | None = None,
) -> dict[int, tuple[str, tuple[int, int, int]]]:
    """``FreeSurferColorLUT.txt`` as ``{id: (name, rgb)}``; ``{}`` if it cannot be found.

    Defaults to ``$FREESURFER_HOME/FreeSurferColorLUT.txt``. Without it a
    volume atlas still answers, with numbers instead of names.
    """
    if path is None:
        home = os.environ.get("FREESURFER_HOME")
        if not home:
            return {}
        path = Path(home) / "FreeSurferColorLUT.txt"
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[int, tuple[str, tuple[int, int, int]]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or line.lstrip().startswith("#"):
            continue
        try:
            out[int(parts[0])] = (parts[1], (int(parts[2]), int(parts[3]), int(parts[4])))
        except ValueError:
            continue
    return out


def load_subject(
    subject_dir: str | os.PathLike, hemis: tuple[str, ...] = ("lh", "rh"), **kwargs
) -> dict[str, Hemisphere]:
    """Read every requested hemisphere present under ``subject_dir/surf``."""
    surf = Path(subject_dir) / "surf"
    if not surf.is_dir():
        raise FileNotFoundError(
            f"{subject_dir} has no surf/ directory; is it a FreeSurfer subject?"
        )
    return {
        h: load_hemisphere(subject_dir, h, **kwargs)
        for h in hemis
        if (surf / f"{h}.white").exists()
    }
