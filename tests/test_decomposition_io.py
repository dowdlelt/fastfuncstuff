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
