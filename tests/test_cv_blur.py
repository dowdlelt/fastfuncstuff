"""Tests for -cv_blur: blur the selection stage, fit the final model unblurred.

The property that matters is that the blur is *contained*. If it leaks into the
final fit, the saved betas are smoothed without the user asking, which is the
one thing -cv_blur exists to avoid (as opposed to -do_blur, which smooths
everything on purpose).
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import blur_masked_data

TR = 1.0
N_TP = 80
SHAPE = (6, 6, 3)


class TestBlurMaskedData:
    def _volume_mask(self, keep):
        mask = np.zeros(SHAPE, dtype=bool)
        mask[keep] = True
        return mask.reshape(-1)

    def test_constant_data_survives_the_mask_edge(self):
        """Normalized convolution: a constant field must blur to itself.

        Without dividing by the identically-blurred mask, zero padding drags
        every edge voxel toward 0 — indistinguishable from signal dropout to
        whatever criterion reads the result.
        """
        mask_flat = self._volume_mask((slice(1, 5), slice(1, 5), slice(None)))
        n_vox = int(mask_flat.sum())
        data = torch.full((n_vox, 10), 7.0)

        out = blur_masked_data(
            data,
            fwhm_mm=5.0,
            volume_shape=SHAPE,
            voxel_sizes=(2.0, 2.0, 2.0),
            mask_flat=mask_flat,
            run_starts=[0],
            device=torch.device("cpu"),
            verbose=False,
        )
        assert torch.allclose(out, data, atol=1e-4)

    def test_constant_survives_fov_edge_without_a_mask(self):
        n_vox = int(np.prod(SHAPE))
        data = torch.full((n_vox, 4), -3.0)
        out = blur_masked_data(
            data,
            fwhm_mm=6.0,
            volume_shape=SHAPE,
            voxel_sizes=(2.0, 2.0, 2.0),
            mask_flat=None,
            run_starts=[0],
            device=torch.device("cpu"),
            verbose=False,
        )
        assert torch.allclose(out, data, atol=1e-4)

    def test_blur_reduces_spatial_variance(self):
        rng = np.random.default_rng(0)
        n_vox = int(np.prod(SHAPE))
        data = torch.from_numpy(rng.standard_normal((n_vox, 20)).astype(np.float32))
        out = blur_masked_data(
            data,
            fwhm_mm=4.0,
            volume_shape=SHAPE,
            voxel_sizes=(2.0, 2.0, 2.0),
            mask_flat=None,
            run_starts=[0],
            device=torch.device("cpu"),
            verbose=False,
        )
        assert out.std(dim=0).mean() < data.std(dim=0).mean()

    def test_input_is_not_modified(self):
        # The caller still needs the unblurred data for the final fit.
        rng = np.random.default_rng(1)
        n_vox = int(np.prod(SHAPE))
        data = torch.from_numpy(rng.standard_normal((n_vox, 12)).astype(np.float32))
        before = data.clone()
        blur_masked_data(
            data,
            fwhm_mm=4.0,
            volume_shape=SHAPE,
            voxel_sizes=(2.0, 2.0, 2.0),
            mask_flat=None,
            run_starts=[0, 6],
            device=torch.device("cpu"),
            verbose=False,
        )
        assert torch.equal(data, before)

    def test_runs_are_blurred_independently_in_space_only(self):
        """Blur is spatial: a run of zeros must stay zero, whatever neighbours it."""
        n_vox = int(np.prod(SHAPE))
        data = torch.zeros(n_vox, 20)
        data[:, :10] = 5.0  # run 0 hot, run 1 empty
        out = blur_masked_data(
            data,
            fwhm_mm=5.0,
            volume_shape=SHAPE,
            voxel_sizes=(2.0, 2.0, 2.0),
            mask_flat=None,
            run_starts=[0, 10],
            device=torch.device("cpu"),
            verbose=False,
        )
        assert torch.allclose(out[:, 10:], torch.zeros_like(out[:, 10:]), atol=1e-6)

    def test_mask_size_mismatch_is_caught(self):
        mask_flat = self._volume_mask((slice(1, 5), slice(1, 5), slice(None)))
        data = torch.zeros(int(mask_flat.sum()) + 3, 5)
        with pytest.raises(ValueError, match="mask selects"):
            blur_masked_data(
                data,
                fwhm_mm=4.0,
                volume_shape=SHAPE,
                voxel_sizes=(2.0, 2.0, 2.0),
                mask_flat=mask_flat,
                run_starts=[0],
                device=torch.device("cpu"),
                verbose=False,
            )


class TestFlagWiring:
    @pytest.mark.parametrize("module", ["denoise", "hrfopt"])
    def test_both_blur_axes_exist(self, module):
        import importlib

        parser = importlib.import_module(f"fastfuncstuff.cli.{module}").create_parser()
        dests = {a.dest for a in parser._actions}
        assert "cv_blur" in dests
        assert "do_blur" in dests  # -cv_blur does not replace it

        base = [
            "-input",
            "run1.nii.gz",
            "run2.nii.gz",
            "-onsets",
            "cond.txt",
            "-durations",
            "2.0",
            "-tr",
            "2.0",
            "-prefix",
            "out",
        ]
        args = parser.parse_args(base + ["-cv_blur", "4"])
        assert args.cv_blur == 4.0
        assert args.do_blur is None


@pytest.mark.slow
def test_cv_blur_does_not_touch_the_final_betas(monkeypatch, tmp_path):
    """With -max_comps 0, nothing from the selection stage enters the final model.

    So a -cv_blur run and a plain run must produce bit-identical betas. If the
    blur leaked past selection, they would not. (At max_comps > 0 they legitimately
    differ: the chosen components are themselves extracted from blurred data.)
    """
    from fastfuncstuff.cli import denoise as denoise_cli

    rng = np.random.default_rng(3)
    onsets = {"faces": [10.0, 30.0], "houses": [20.0, 40.0]}
    inputs = []
    for run in range(2):
        vol = 100.0 + 3.0 * rng.standard_normal((*SHAPE, N_TP))
        for times in onsets.values():
            for onset in times:
                t0 = int(round(onset / TR)) + 2
                vol[:2, :2, :, t0 : t0 + 4] += 8.0
        path = tmp_path / f"run{run + 1}.nii.gz"
        img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
        img.header["pixdim"][4] = TR
        nib.save(img, str(path))
        inputs.append(str(path))

    onset_files = []
    for cond, times in onsets.items():
        path = tmp_path / f"{cond}.txt"
        row = " ".join(f"{t:.1f}" for t in times)
        path.write_text(f"{row}\n{row}\n")
        onset_files.append(str(path))

    def _run(prefix, extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ffs_denoise",
                "-input",
                *inputs,
                "-onsets",
                *onset_files,
                "-durations",
                "4.0",
                "-tr",
                str(TR),
                "-prefix",
                str(tmp_path / prefix),
                "-device",
                "cpu",
                "-max_comps",
                "0",  # no PCs can be selected, so nothing from the search reaches the fit
                "-min_noise_voxels",
                "10",
                "-plots",
                "no",
                "-single_trials",
                *extra,
            ],
        )
        denoise_cli.main()
        return nib.load(str(tmp_path / f"{prefix}_single_trial_betas.nii.gz")).get_fdata()

    plain = _run("plain", [])
    blurred = _run("blurred", ["-cv_blur", "5"])

    assert np.array_equal(plain, blurred)
    # ...and the blur really did something upstream.
    import json

    meta = json.loads(Path(f"{tmp_path}/blurred_denoise_metadata.json").read_text())
    assert meta["cv_blur_fwhm"] == 5.0
