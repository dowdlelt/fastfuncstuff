"""Merged two-hemisphere meshes: splitting a result back into .lh / .rh files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.cli_utils import split_ribbon_outputs
from fastfuncstuff.io.gifti import (
    load_gifti_data,
    mesh_fingerprint,
    save_gifti_data,
    save_gifti_surface,
)
from fastfuncstuff.processing.ribbon_sample import (
    MERGED_HEMIS_KEY,
    default_master,
    group_hemisphere_files,
    hemisphere_of,
    merge_hemisphere_runs,
    split_merged_surface,
)


def _triangle(offset: float):
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32) + offset
    return vertices, np.array([[0, 1, 2]], np.int32)


@pytest.fixture
def merged_bucket(tmp_path):
    """A two-column bucket on a merged mesh of two 3-vertex hemispheres."""
    bounds = []
    at = 0
    for hemi, offset in (("lh", 0.0), ("rh", 10.0)):
        vertices, faces = _triangle(offset)
        path = tmp_path / f"geom.native.{hemi}.midthickness.surf.gii"
        save_gifti_surface(path, vertices, faces)
        bounds.append(
            {
                "hemi": hemi,
                "start": at,
                "stop": at + 3,
                "mesh_fingerprint": mesh_fingerprint(faces, 3),
                "geometry": path.name,
            }
        )
        at += 3

    import nibabel as nib

    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    arrays = [
        nib.gifti.GiftiDataArray(
            np.ascontiguousarray(values[:, k]),
            datatype="NIFTI_TYPE_FLOAT32",
            meta={"Name": name},
        )
        for k, name in enumerate(("x", "r2"))
    ]
    meta = {MERGED_HEMIS_KEY: json.dumps(bounds), "mesh_fingerprint": "merged", "other": "kept"}
    path = tmp_path / "prf.func.gii"
    nib.save(nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(meta)), path)
    return path, bounds, values


def test_split_gives_each_hemisphere_its_own_rows_fingerprint_and_geometry(merged_bucket):
    """The merged file says where the hemispheres are, so the split needs nothing else."""
    path, bounds, values = merged_bucket

    written = split_merged_surface(path)

    assert [p.rsplit("/", 1)[-1] for p in written] == ["prf.lh.func.gii", "prf.rh.func.gii"]
    for part, out_path in zip(bounds, written, strict=True):
        data, meta = load_gifti_data(out_path)
        np.testing.assert_array_equal(data, values[part["start"] : part["stop"]])
        assert meta["mesh_fingerprint"] == part["mesh_fingerprint"]
        assert meta["geometry"] == part["geometry"]
        assert meta["hemisphere"] == part["hemi"]
        assert meta["other"] == "kept"
        assert MERGED_HEMIS_KEY not in meta


def test_split_keeps_the_bucket_labels(merged_bucket):
    """A pRF bucket is only readable by column name; the split must not drop them."""
    import nibabel as nib

    path, _, _ = merged_bucket

    written = split_merged_surface(path)

    image = nib.load(written[0])
    assert [dict(a.meta)["Name"] for a in image.darrays] == ["x", "r2"]


def test_split_refuses_a_file_that_is_not_on_a_merged_mesh(tmp_path):
    """Without the metadata there is no honest guess at where the midline is."""
    from fastfuncstuff.io.gifti import save_gifti_data

    path = tmp_path / "plain.shape.gii"
    save_gifti_data(path, np.zeros(6, np.float32), {"mesh_fingerprint": "whatever"})

    with pytest.raises(ValueError, match="merged mesh"):
        split_merged_surface(path)


def test_split_outputs_pass_leaves_unmerged_files_alone(merged_bucket, tmp_path):
    """The post-pass splits what landed on the merged mesh and nothing else."""
    from fastfuncstuff.io.gifti import save_gifti_data

    path, _, _ = merged_bucket
    plain = tmp_path / "prf_screen.shape.gii"
    save_gifti_data(plain, np.zeros(6, np.float32), {"mesh_fingerprint": "whatever"})

    written = split_ribbon_outputs(str(tmp_path / "prf"), verbose=False)

    assert sorted(p.rsplit("/", 1)[-1] for p in written) == ["prf.lh.func.gii", "prf.rh.func.gii"]
    assert not path.exists()  # replaced by its two halves
    assert plain.exists()


def test_default_master_names_the_candidates_it_looked_for(tmp_path):
    """A subject without an anatomy in surface space has to be told which one to use."""
    (tmp_path / "surf").mkdir()

    with pytest.raises(FileNotFoundError, match="SUMA/brain.nii.gz"):
        default_master(tmp_path)


def test_geometry_merge_offsets_the_faces_and_records_each_hemisphere(tmp_path):
    """The merged mesh is two disconnected components, in hemisphere order."""
    from fastfuncstuff.io.gifti import load_gifti_surface
    from fastfuncstuff.processing.ribbon_sample import _geometry

    for hemi, offset in (("lh", 0.0), ("rh", 10.0)):
        vertices, faces = _triangle(offset)
        save_gifti_surface(tmp_path / f"geom.native.{hemi}.midthickness.surf.gii", vertices, faces)
    merged_path = str(tmp_path / "geom.native.midthickness.surf.gii")

    bounds, sizes, fingerprint = _geometry(
        str(tmp_path / "geom"), "native", ("lh", "rh"), merged_path, tmp_path
    )

    assert sizes == [3, 3]
    assert [(b["hemi"], b["start"], b["stop"]) for b in bounds] == [("lh", 0, 3), ("rh", 3, 6)]
    vertices, faces, meta = load_gifti_surface(merged_path)
    assert len(vertices) == 6
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5]])
    assert meta["mesh_fingerprint"] == fingerprint


def test_reuse_requires_the_same_settings_and_a_source_no_newer(tmp_path):
    """Sampling is minutes per run, but only an identical sampling may be reused."""
    import json
    import os

    from fastfuncstuff.processing.ribbon_sample import _reusable

    source = tmp_path / "run.nii.gz"
    source.write_bytes(b"")
    merged = tmp_path / "run-01.native.func.gii"
    merged.write_bytes(b"")
    sidecar = tmp_path / "run-01.native.sample.json"
    settings = {"subject": "s", "mesh": "native", "hemis": ["lh"], "depths": [0.5], "sampling": "f"}
    sidecar.write_text(json.dumps({**settings, "source": str(source)}))

    assert _reusable(str(merged), sidecar, str(source), settings)
    assert not _reusable(str(merged), sidecar, str(source), {**settings, "depths": [0.1, 0.9]})

    # A source touched after the sampling is new data, whatever the settings say.
    os.utime(source, (merged.stat().st_mtime + 10, merged.stat().st_mtime + 10))
    assert not _reusable(str(merged), sidecar, str(source), settings)


# --------------------------------------------------------------------------------
# Pairing hemispheres that are already on a surface
# --------------------------------------------------------------------------------


def test_hemisphere_of_reads_the_filename_and_refuses_ambiguity():
    assert hemisphere_of("ses-1.lh.func.gii")[0] == "lh"
    # A parent directory is not a claim about the file in it.
    assert hemisphere_of("/data/lh/sub.rh.func.gii")[0] == "rh"
    # Delimited, so a hemisphere is never read out of the middle of a word.
    assert hemisphere_of("x.lhrh.gii") is None
    assert hemisphere_of("Alh.func.gii") is None
    assert hemisphere_of("blah.func.gii") is None
    # Two in one name has no single answer; taking the first mislabels a hemisphere.
    assert hemisphere_of("sub.lh.rh.func.gii") is None
    # The key is what the two hemispheres share, so they group together and
    # different runs do not.
    assert hemisphere_of("r1.lh.gii")[1] == hemisphere_of("r1.rh.gii")[1]
    assert hemisphere_of("r1.lh.gii")[1] != hemisphere_of("r2.lh.gii")[1]


def _hemi_runs(tmp_path, n_runs=2, n_vertices=3, n_times=4, hemis=("lh", "rh")):
    """``n_runs`` runs written per hemisphere, as ffs_nwarp -surf writes them."""
    written = {}
    for hemi, offset in zip(hemis, (0.0, 10.0), strict=False):
        vertices, faces = _triangle(offset)
        geom = tmp_path / f"geom.{hemi}.midthickness.surf.gii"
        save_gifti_surface(geom, vertices, faces)
        for run in range(1, n_runs + 1):
            path = tmp_path / f"run-{run:02d}.native.{hemi}.func.gii"
            data = np.arange(n_vertices * n_times, dtype=np.float32).reshape(
                n_vertices, n_times
            ) + (1000 * run + (0 if hemi == "lh" else 100))
            save_gifti_data(
                path, data, {"geometry": geom.name, "TR_seconds": "1.5"}, time_series=True
            )
            written.setdefault(hemi, []).append(str(path))
    return written


def test_group_pairs_one_listed_hemisphere_with_the_file_beside_it(tmp_path):
    files = _hemi_runs(tmp_path)
    grouped, hemis = group_hemisphere_files(files["lh"])
    assert hemis == ["lh", "rh"]
    assert [[Path(p).name for p in run] for run in grouped] == [
        ["run-01.native.lh.func.gii", "run-01.native.rh.func.gii"],
        ["run-02.native.lh.func.gii", "run-02.native.rh.func.gii"],
    ]
    # Listing both, interleaved, must group to exactly the same thing.
    both = [p for pair in zip(files["lh"], files["rh"], strict=True) for p in pair]
    assert group_hemisphere_files(both) == (grouped, hemis)
    # Run order follows the order given: -stimulus is positional against it.
    reversed_grouped, _ = group_hemisphere_files(files["lh"][::-1])
    assert reversed_grouped == grouped[::-1]


def test_group_declines_what_is_not_a_pair(tmp_path):
    """Not-a-pair is the ordinary answer, not an error: a merged dataset, a volume,
    or a single hemisphere someone means to fit alone."""
    files = _hemi_runs(tmp_path, n_runs=1)
    assert group_hemisphere_files(["run.nii.gz"]) is None
    assert group_hemisphere_files(["merged.func.gii"]) is None
    assert group_hemisphere_files([]) is None
    # A dataset with no rh at all is a thing that happens; pairing is a convenience,
    # so it declines and the caller fits the one hemisphere it was given.
    Path(files["rh"][0]).unlink()
    assert group_hemisphere_files(files["lh"]) is None


def test_group_refuses_hemispheres_listed_unevenly(tmp_path):
    """Both hemispheres named, but not for every run: silently fitting the ones
    that pair would drop a run the user asked for."""
    files = _hemi_runs(tmp_path, n_runs=2)
    with pytest.raises(FileNotFoundError, match="some runs and one for others"):
        group_hemisphere_files([*files["lh"], files["rh"][0]])


def test_merge_stacks_the_hemispheres_and_describes_where_they_landed(tmp_path):
    files = _hemi_runs(tmp_path, n_runs=2, n_vertices=3, n_times=4)
    grouped, hemis = group_hemisphere_files(files["lh"])
    sample = merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)

    assert sample.n_vertices == 6
    assert [b["hemi"] for b in sample.hemis] == ["lh", "rh"]
    assert [(b["start"], b["stop"]) for b in sample.hemis] == [(0, 3), (3, 6)]
    data, meta = load_gifti_data(sample.runs[0])
    assert data.shape == (6, 4)
    lh = load_gifti_data(files["lh"][0])[0]
    rh = load_gifti_data(files["rh"][0])[0]
    np.testing.assert_array_equal(data[:3], lh)
    np.testing.assert_array_equal(data[3:], rh)
    # The TR has to survive: the fit's design is built in seconds.
    assert meta["TR_seconds"] == "1.5"
    # And the result must be splittable afterwards, which is the whole point.
    assert MERGED_HEMIS_KEY in meta
    assert Path(sample.geometry).is_file()


def test_the_merged_result_splits_back_to_the_hemispheres_it_came_from(tmp_path):
    """The round trip is the contract: fit one dataset, get one file per hemisphere."""
    files = _hemi_runs(tmp_path, n_runs=1, n_vertices=3, n_times=5)
    grouped, hemis = group_hemisphere_files(files["lh"])
    sample = merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)
    data, meta = load_gifti_data(sample.runs[0])
    save_gifti_data(tmp_path / "m" / "fit.func.gii", data, meta, time_series=False)

    parts = split_merged_surface(tmp_path / "m" / "fit.func.gii")
    assert [Path(p).name for p in parts] == ["fit.lh.func.gii", "fit.rh.func.gii"]
    back = np.concatenate([load_gifti_data(p)[0] for p in parts])
    np.testing.assert_array_equal(back, data)
    # Each half points at its own hemisphere's geometry, not the merged mesh.
    for part, hemi in zip(parts, hemis, strict=True):
        part_meta = load_gifti_data(part)[1]
        assert part_meta["hemisphere"] == hemi
        assert (
            Path(part)
            .parent.joinpath(part_meta["geometry"])
            .resolve()
            .name.startswith(f"geom.{hemi}")
        )


def test_merge_without_findable_geometry_still_merges(tmp_path):
    """A mesh is only needed by the operations that are geometric. Losing the
    geometry pointer must not cost the fit, which is per-vertex."""
    files = _hemi_runs(tmp_path, n_runs=1)
    for hemi in ("lh", "rh"):
        Path(tmp_path / f"geom.{hemi}.midthickness.surf.gii").unlink()
    grouped, hemis = group_hemisphere_files(files["lh"])
    sample = merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)
    assert sample.geometry == ""
    assert sample.n_vertices == 6
    assert load_gifti_data(sample.runs[0])[0].shape[0] == 6


def test_merge_reuses_a_merged_run_until_a_source_changes(tmp_path):
    """Merging is cheap but not free, and a second pass over the same data (another
    model, another HRF setting) should not redo it."""
    files = _hemi_runs(tmp_path, n_runs=1)
    grouped, hemis = group_hemisphere_files(files["lh"])
    first = merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)
    stamp = Path(first.runs[0]).stat().st_mtime_ns
    merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)
    assert Path(first.runs[0]).stat().st_mtime_ns == stamp  # not rewritten

    os.utime(files["lh"][0], (stamp / 1e9 + 10, stamp / 1e9 + 10))
    merge_hemisphere_runs(grouped, hemis, out_dir=tmp_path / "m", verb=0)
    assert Path(first.runs[0]).stat().st_mtime_ns != stamp  # a newer source redoes it
