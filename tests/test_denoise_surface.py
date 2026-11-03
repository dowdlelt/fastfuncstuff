"""Surface denoising outputs retain mesh metadata and use GIfTI files."""

import numpy as np
import pytest
import torch

from fastfuncstuff.cli.denoise import _denoise_output_extension, save_model_fit_outputs
from fastfuncstuff.glm.outputs import save_single_trial_results
from fastfuncstuff.io.gifti import load_gifti_data, set_surface_meta


def _header():
    import nibabel as nib

    header = nib.Nifti2Header()
    set_surface_meta(header, {"mesh_fingerprint": "V3_F1_example", "geometry": "mesh.surf.gii"})
    return header


def test_surface_extension_and_unsupported_volume_operations():
    files = ["left.func.gii", "right.func.gii"]
    assert _denoise_output_extension(files, "fit") == ".func.gii"
    assert _denoise_output_extension(files, "fit.shape.gii") == ".shape.gii"
    with pytest.raises(ValueError, match="mix surface and volume"):
        _denoise_output_extension([files[0], "run.nii.gz"], "fit")
    with pytest.raises(ValueError, match="automask"):
        _denoise_output_extension(files, "fit", automask=True)
    with pytest.raises(ValueError, match="cv_blur"):
        _denoise_output_extension(files, "fit", cv_blur=2.0)
    # The domain rule itself lives in cli_utils.domain_prefix, shared with
    # ffs_pyrf and ffs_varpart; what is tested here is that denoise routes to it.
    with pytest.raises(ValueError, match=r"names a NIfTI file.*on a surface"):
        _denoise_output_extension(files, "fit.nii.gz")
    with pytest.raises(ValueError, match=r"names a GIfTI file.*on a volume"):
        _denoise_output_extension(["run.nii.gz"], "fit.func.gii")


def test_surface_model_fit_bucket_preserves_mesh_and_labels(tmp_path):
    from types import SimpleNamespace

    fit = SimpleNamespace(
        betas=torch.tensor([[1.0], [2.0], [3.0]]),
        tstats=torch.tensor([[4.0], [5.0], [6.0]]),
    )
    paths = save_model_fit_outputs(
        fit,
        str(tmp_path / "fit"),
        (3, 1, 1),
        np.eye(4),
        "initial",
        condition_labels=["trial"],
        nifti_header=_header(),
        nii_ext=".func.gii",
    )
    data, meta = load_gifti_data(paths["initial_bucket"])
    assert data.shape == (3, 2)
    np.testing.assert_array_equal(data[:, 1], [4, 5, 6])
    assert meta["mesh_fingerprint"] == "V3_F1_example"


def test_surface_single_trial_maps_use_requested_extension(tmp_path):
    paths = save_single_trial_results(
        betas=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        xval_r2=torch.tensor([0.1, 0.2, 0.3]),
        trial_labels=["a", "b"],
        trial_condition_ids=torch.tensor([0, 0]),
        trial_run_ids=torch.tensor([0, 1]),
        condition_labels=["condition"],
        output_prefix=str(tmp_path / "fit"),
        volume_shape=(3, 1, 1),
        affine=np.eye(4),
        nifti_header=_header(),
        nii_ext=".func.gii",
    )
    for key in ("single_trial_betas", "single_trial_xval", "single_trial_condition_betas"):
        path = paths[key]
        assert path.endswith(".func.gii")
        _, meta = load_gifti_data(path)
        assert meta["mesh_fingerprint"] == "V3_F1_example"
