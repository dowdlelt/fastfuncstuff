"""Regression tests for simulation output utilities."""

import nibabel as nib
import numpy as np
import torch

from fastfuncsim.simulation.core import (
    save_simulation_outputs,
    write_afni_onset_files,
    write_nifti_files,
)


def _make_onset(run_length: int, events: dict, n_conditions: int | None = None) -> torch.Tensor:
    """Build a binary onset matrix of shape (run_length, n_conditions)."""
    if n_conditions is None:
        if events:
            n_conditions = max(events.keys()) + 1
        else:
            n_conditions = 1
    onset = torch.zeros(run_length, n_conditions)
    for cond_idx, timepoints in events.items():
        if len(timepoints) == 0:
            continue
        indices = torch.tensor(timepoints, dtype=torch.long)
        onset[indices, cond_idx] = 1.0
    return onset


def test_write_afni_onset_files_handles_multiple_runs_and_empty_condition(tmp_path):
    tr = 2.0
    run1 = _make_onset(6, {0: [0, 4], 1: [1]}, n_conditions=2)
    run2 = _make_onset(6, {0: [2]}, n_conditions=2)  # Condition 2 has no events -> "*"

    files = write_afni_onset_files([run1, run2], tr=tr, output_dir=tmp_path, prefix="onsets")

    assert len(files) == 2
    cond1_path, cond2_path = files
    assert cond1_path.name == "onsets_condition1.txt"
    assert cond2_path.name == "onsets_condition2.txt"

    cond1_lines = cond1_path.read_text().strip().splitlines()
    cond2_lines = cond2_path.read_text().strip().splitlines()

    assert cond1_lines[0] == "0.00 8.00"
    assert cond1_lines[1] == "4.00"

    assert cond2_lines[0] == "2.00"
    assert cond2_lines[1] == "*"


def test_write_nifti_files_roundtrip_metadata(tmp_path):
    tr = 1.5
    voxel_size = (2.0, 3.0, 4.0)
    run_shape = (2, 2, 1, 4)

    run1 = torch.arange(int(np.prod(run_shape)), dtype=torch.float32).reshape(run_shape)
    run2 = torch.full(run_shape, 7.0, dtype=torch.float32)

    files = write_nifti_files([run1, run2], tr=tr, output_dir=tmp_path, prefix="run", voxel_size=voxel_size)

    assert len(files) == 2
    first_img = nib.load(str(files[0]))

    assert first_img.shape == run_shape
    header = first_img.header
    assert np.isclose(header["pixdim"][1], voxel_size[0])
    assert np.isclose(header["pixdim"][2], voxel_size[1])
    assert np.isclose(header["pixdim"][3], voxel_size[2])
    assert np.isclose(header["pixdim"][4], tr)
    assert first_img.get_data_dtype() == np.float32


def test_save_simulation_outputs_creates_expected_folder_structure(tmp_path):
    tr = 2.0
    run_shape = (2, 2, 1, 3)
    data_list = [
        torch.linspace(0, 1, steps=int(np.prod(run_shape)), dtype=torch.float32).reshape(run_shape),
        torch.ones(run_shape, dtype=torch.float32),
    ]

    onsets_list = [
        _make_onset(run_shape[-1], {0: [0, 2]}, n_conditions=2),
        _make_onset(run_shape[-1], {1: [1]}, n_conditions=2),
    ]

    metadata = {"betas": [1.0, 2.0], "noise_level": 0.5}

    info = save_simulation_outputs(
        data_list=data_list,
        onsets_list=onsets_list,
        tr=tr,
        output_dir=tmp_path,
        label="unit",
        metadata=metadata,
        verbose=False,
    )

    sim_dir = tmp_path / "simulation_unit"
    assert info["output_dir"] == sim_dir
    assert sim_dir.exists()

    onset_files = info["onset_files"]
    nifti_files = info["nifti_files"]
    metadata_file = info["metadata_file"]

    assert len(onset_files) == 2
    assert all(f.parent == sim_dir for f in onset_files)
    assert all(f.name.startswith("onsets_condition") for f in onset_files)

    assert len(nifti_files) == len(data_list)
    assert all(f.parent == sim_dir for f in nifti_files)
    assert nifti_files[0].name == "run01.nii.gz"

    meta_text = metadata_file.read_text()
    assert "Simulation Label: unit" in meta_text
    assert "Number of runs: 2" in meta_text
    assert "betas: [1.0, 2.0]" in meta_text

    saved_img = nib.load(str(nifti_files[0]))
    assert np.isclose(saved_img.header["pixdim"][4], tr)