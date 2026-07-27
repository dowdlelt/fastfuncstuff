"""Targeted tests for decomposition I/O utilities."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.decomposition.io import (
    load_component_maps,
    load_timeseries,
    safe_relative_symlink,
    save_component_maps,
    save_decomposition_results,
    save_masked_component_maps_4d,
    save_timeseries,
    write_melodic_compat_outputs,
)


def test_save_masked_component_maps_4d_unmasked_size_mismatch(tmp_path):
    components = np.ones((2, 5), dtype=np.float32)
    affine = np.eye(4)

    with pytest.raises(ValueError, match="Component size does not match"):
        save_masked_component_maps_4d(
            components_kv=components,
            mask3d=None,
            shape3d=(2, 2, 1),
            affine=affine,
            out_file=tmp_path / "bad_maps.nii.gz",
        )


def test_safe_relative_symlink_replaces_existing(tmp_path):
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "nested" / "b.txt"
    link = tmp_path / "links" / "current.txt"

    target_a.write_text("A")
    target_b.parent.mkdir(parents=True, exist_ok=True)
    target_b.write_text("B")

    first = safe_relative_symlink(target_a, link)
    assert first.is_symlink()
    assert not Path(first.readlink()).is_absolute()

    second = safe_relative_symlink(target_b, link)
    assert second.is_symlink()
    assert not Path(second.readlink()).is_absolute()
    assert second.resolve() == target_b.resolve()


def test_save_and_load_timeseries_1d_roundtrip(tmp_path):
    ts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = ["IC_0", "IC_1"]
    out_file = tmp_path / "mix.1D"

    save_timeseries(timeseries=ts, output_file=out_file, labels=labels)
    loaded, loaded_labels = load_timeseries(out_file)

    assert loaded.shape == ts.shape
    assert np.allclose(loaded, ts)
    assert loaded_labels == labels


def test_save_and_load_component_maps_roundtrip(tmp_path):
    mask_data = np.array([[[1], [0]], [[1], [1]]], dtype=np.uint8)
    mask_img = nib.Nifti1Image(mask_data, np.eye(4))
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(mask_img, mask_file)

    components = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    labels = ["IC_A", "IC_B"]
    map_file = tmp_path / "maps.nii.gz"

    save_component_maps(components, mask_file, map_file, labels=labels)
    loaded_components, loaded_labels = load_component_maps(map_file, mask_file)

    assert loaded_components.shape == components.shape
    assert np.allclose(loaded_components, components)
    assert loaded_labels == ["Component_0", "Component_1"]


def test_save_timeseries_nifti_requires_tr_or_reference(tmp_path):
    ts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Must provide either 'tr' or 'reference_file'"):
        save_timeseries(timeseries=ts, output_file=tmp_path / "mix.nii.gz")


def test_save_decomposition_results_creates_expected_outputs(tmp_path):
    mask_data = np.array([[[1], [0]], [[1], [1]]], dtype=np.uint8)
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_file)

    components = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    timeseries = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    outputs = save_decomposition_results(
        components=components,
        timeseries=timeseries,
        mask_file=mask_file,
        output_prefix=tmp_path / "ica_run",
        method="ICA",
    )

    assert outputs["maps"].exists()
    assert outputs["timeseries_1D"].exists()
    assert outputs["maps"].name == "ica_run_maps.nii.gz"
    assert outputs["timeseries_1D"].name == "ica_run_timeseries.1D"


@pytest.mark.parametrize("n_t", [212, 423])
def test_melodic_compat_ftmix_matches_fsleyes_frequency_axis(tmp_path, n_t):
    """melodic_FTmix must have ceil(T/2) rows.

    FSLeyes builds the power-spectrum x-axis as rfftfreq(T + T%2)[1:]; an rfft
    on the raw odd length yields one row fewer and the plot fails to draw.
    """
    n_k, shape3d = 4, (2, 2, 2)
    n_vox = int(np.prod(shape3d))
    rng = np.random.default_rng(0)
    mixing = rng.standard_normal((n_t, n_k))

    maps_file = tmp_path / "ica_maps.nii.gz"
    save_masked_component_maps_4d(
        components_kv=rng.standard_normal((n_k, n_vox)).astype(np.float32),
        mask3d=None,
        shape3d=shape3d,
        affine=np.eye(4),
        out_file=maps_file,
    )
    tcs_file = tmp_path / "ica_timecourses.1D"
    np.savetxt(tcs_file, mixing, fmt="%.6f")

    scree = np.full(n_k, 1.0 / n_k)
    write_melodic_compat_outputs(
        compat_dir=tmp_path / "out.ica",
        maps_file=maps_file,
        zmaps_file=None,
        timecourse_file=tcs_file,
        pca_scree_ratio=scree,
        component_explained_share_pct=scree * 100.0,
        component_total_share_pct=scree * 100.0,
        mixing_np=mixing,
        mask3d=None,
        mean3d=np.ones(shape3d, dtype=np.float32),
        shape3d=shape3d,
        affine=np.eye(4),
    )

    ftmix = np.loadtxt(tmp_path / "out.ica" / "melodic_FTmix")
    expected = len(np.fft.rfftfreq(n_t + (n_t % 2))[1:])
    assert ftmix.shape == (expected, n_k)

    # eigenvalues_percent: single row of cumulative fraction ending at 1.0
    with open(tmp_path / "out.ica" / "eigenvalues_percent") as fh:
        assert len([ln for ln in fh if ln.strip()]) == 1
    ev = np.loadtxt(tmp_path / "out.ica" / "eigenvalues_percent")
    assert ev.shape == (n_k,)
    assert ev[-1] == pytest.approx(1.0)
    assert np.all(np.diff(ev) >= 0)
