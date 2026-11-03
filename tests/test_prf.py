"""Tests for the CSS pRF primitives and the ffs_pyrf input helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.cli.pyrf import (
    _load_png_directory,
    _load_stimulus_run,
    _load_stimulus_sources,
)
from fastfuncstuff.design.prf import (
    PRFGrid,
    PRFGridFit,
    PRFRefinementConfig,
    _css_prediction,
    _css_prediction_and_derivatives,
    choose_aperture_bin_factor,
    downsample_aperture,
    fit_prf_loro,
    fit_prf_supergrid,
    gaussian_receptive_fields,
    make_analyzeprf_grid,
    prf_parameter_maps,
    refine_prf_supergrid,
)


def test_png_stimulus_directory_uses_natural_frame_order(tmp_path, monkeypatch):
    """Frame 2 must precede frame 10 regardless of filesystem order."""
    for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
        (tmp_path / name).touch()

    values = {"frame_1.png": 1.0, "frame_2.png": 2.0, "frame_10.png": 10.0}

    def fake_imread(path):
        return np.full((2, 3), values[path.name], dtype=np.float32)

    monkeypatch.setattr("fastfuncstuff.cli.pyrf.mpimg.imread", fake_imread)

    movie = _load_png_directory(tmp_path)

    assert movie.shape == (2, 3, 3)
    np.testing.assert_array_equal(movie[0, 0], np.array([1.0, 2.0, 10.0]))


def test_stimulus_nifti_drops_the_singleton_slice_axis(tmp_path):
    """A row x column x 1 x time aperture NIfTI loads as time-by-pixel samples."""
    import nibabel as nib

    movie = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    nib.save(nib.Nifti1Image(movie[:, :, None, :], np.eye(4)), tmp_path / "stim.nii.gz")

    frames, shape = _load_stimulus_run(str(tmp_path / "stim.nii.gz"))

    assert shape == (2, 3)
    assert frames.shape == (4, 6)
    np.testing.assert_allclose(frames.numpy(), np.moveaxis(movie, -1, 0).reshape(4, 6))


def test_aperture_bin_factor_prefers_an_exact_divisor_near_the_target():
    """1080 must bin by 10 to 108, not resample unevenly to exactly 100."""
    assert choose_aperture_bin_factor(1080, 100) == 10
    assert choose_aperture_bin_factor(768, 100) == 8
    assert choose_aperture_bin_factor(100, 100) == 1
    assert choose_aperture_bin_factor(64, 100) == 1


def test_downsample_aperture_block_averages_fractional_coverage():
    """A binary aperture becomes the mean coverage of each block."""
    movie = torch.zeros(8, 8, 2)
    movie[:4, :, 0] = 1.0
    movie[:2, :2, 1] = 1.0

    binned, shape = downsample_aperture(movie, 4)

    assert shape == (4, 4)
    torch.testing.assert_close(binned[:, 0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert binned[0, 0, 1].item() == 1.0
    assert binned[0, 1, 1].item() == 0.0


def test_downsample_aperture_resizes_axes_with_no_usable_divisor():
    """A prime-length axis still shrinks rather than staying at full resolution."""
    movie = torch.rand(101, 100, 3)

    binned, shape = downsample_aperture(movie, 10)

    assert shape == (10, 10)
    assert binned.shape == (10, 10, 3)


def test_stimulus_downsample_reaches_the_loader(tmp_path):
    """-stim_downsample shrinks the aperture at load time, before flattening."""
    import nibabel as nib

    movie = np.ones((20, 20, 1, 3), dtype=np.float32)
    nib.save(nib.Nifti1Image(movie, np.eye(4)), tmp_path / "stim.nii.gz")

    frames, shape = _load_stimulus_run(str(tmp_path / "stim.nii.gz"), 5)

    assert shape == (5, 5)
    assert frames.shape == (3, 25)


def test_png_multi_source_splits_frames_at_input_run_boundaries(monkeypatch):
    """One declared PNG source partitions in input-run order using run lengths."""
    frames = torch.arange(30, dtype=torch.float32).reshape(5, 6)

    def fake_load_stimulus_run(path, downsample=0):
        assert path == "all_runs"
        return frames, (2, 3)

    monkeypatch.setattr("fastfuncstuff.cli.pyrf._load_stimulus_run", fake_load_stimulus_run)

    runs, shape = _load_stimulus_sources([["all_runs", "2"]], [2, 3])

    assert shape == (2, 3)
    assert [run.shape for run in runs] == [(2, 6), (3, 6)]
    torch.testing.assert_close(runs[0], frames[:2])
    torch.testing.assert_close(runs[1], frames[2:])


def _synthetic_css_data():
    torch.manual_seed(4)
    stimulus_runs = [
        torch.randint(0, 2, (9, 25), dtype=torch.float32),
        torch.randint(0, 2, (11, 25), dtype=torch.float32),
    ]
    parameters = torch.tensor([[2.8, 3.2, 1.1, 0.65]], dtype=torch.float32)
    hrf = torch.tensor([1.0, 0.4, 0.1], dtype=torch.float32)
    prediction, _ = _css_prediction_and_derivatives(stimulus_runs, parameters, hrf, (5, 5))
    data = (prediction.T * 2.5).contiguous()
    return data, stimulus_runs, parameters, hrf


def test_css_gauss_newton_refines_a_perturbed_supergrid_seed():
    """The analytic GN solver should recover a noiseless CSS signal from a nearby seed."""
    data, stimulus_runs, parameters, hrf = _synthetic_css_data()
    seed_parameters = parameters + torch.tensor([[0.25, -0.2, 0.15, -0.1]])
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(1, dtype=torch.long),
        hrf_index=torch.zeros(1, dtype=torch.long),
        parameters=seed_parameters,
        gain=torch.ones(1),
        correlation=torch.zeros(1),
        r2=torch.zeros(1),
    )

    fit = refine_prf_supergrid(
        data,
        stimulus_runs,
        (5, 5),
        grid_fit,
        hrf.unsqueeze(0),
        [0, 9],
        voxel_chunk_size=1,
        config=PRFRefinementConfig(max_iter=30),
    )

    torch.testing.assert_close(fit.parameters, parameters, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fit.gain, torch.tensor([2.5]), atol=2e-2, rtol=2e-2)
    assert fit.r2.item() > 0.999


def test_loro_prf_uses_fold_local_fit_to_predict_held_out_runs():
    """A one-candidate noiseless model should obtain near-perfect held-out R2."""
    data, stimulus_runs, parameters, hrf = _synthetic_css_data()
    results = fit_prf_loro(
        data,
        stimulus_runs,
        (5, 5),
        PRFGrid(parameters),
        hrf.unsqueeze(0),
        [0, 9],
        candidate_chunk_size=1,
        voxel_chunk_size=1,
        refinement_config=PRFRefinementConfig(max_iter=10),
    )

    assert results.r2.item() > 0.999


def _matlab_reference_field(
    stimulus_shape: tuple[int, int], row: float, column: float, sigma: float
) -> np.ndarray:
    """Transcribe makegaussian2d + placematrix straight from the analyzePRF source.

    analyzePRF builds the Gaussian on a square ``resmx`` grid in unit coordinates
    and then centers it into the real aperture, so a non-square aperture is not
    simply indexed 1..rows / 1..columns.
    """
    rows, columns = stimulus_shape
    extent = max(rows, columns)
    step = 1.0 / extent
    axis = np.linspace(-0.5 + step / 2, 0.5 - step / 2, extent)
    xx, yy = np.meshgrid(axis, -axis)
    row_unit = (-1 / extent) * row + (0.5 + 0.5 / extent)
    column_unit = (1 / extent) * column + (-0.5 - 0.5 / extent)
    unit_sigma = sigma / extent
    square = np.exp(((xx - column_unit) ** 2 + (yy - row_unit) ** 2) / -(2 * unit_sigma**2)) / (
        2 * np.pi * sigma**2
    )
    row_offset = (extent - rows) // 2
    column_offset = (extent - columns) // 2
    return square[row_offset : row_offset + rows, column_offset : column_offset + columns]


def test_receptive_field_matches_analyzeprf_for_non_square_apertures():
    """Non-square apertures must inherit analyzePRF's centered square coordinate frame."""
    for stimulus_shape in [(10, 10), (6, 14), (14, 6)]:
        parameters = torch.tensor([[5.5, 7.2, 2.0, 0.5]], dtype=torch.float64)
        fields = gaussian_receptive_fields(parameters, stimulus_shape)
        expected = _matlab_reference_field(stimulus_shape, 5.5, 7.2, 2.0)
        np.testing.assert_allclose(
            fields.reshape(stimulus_shape).numpy(),
            expected,
            atol=1e-12,
            err_msg=str(stimulus_shape),
        )


def test_supergrid_seeds_are_centered_in_the_square_frame():
    """The zero-eccentricity seed sits at (1+resmx)/2 on both axes, as in MATLAB."""
    grid = make_analyzeprf_grid((6, 14))
    center = (1 + 14) / 2
    torch.testing.assert_close(grid.parameters[0, :2], torch.tensor([center, center]))
    # sigma grid is 1, 2, 4, 8 px (2.^(0:floor(log2(14)))) times sqrt(exponent)
    sigmas = (grid.parameters[:, 2] / grid.parameters[:, 3].sqrt()).unique(sorted=True)
    torch.testing.assert_close(sigmas, torch.tensor([1.0, 2.0, 4.0, 8.0]))


def test_analytic_derivatives_match_finite_differences():
    """Guards the CSS Jacobian, including the non-square coordinate offset."""
    torch.manual_seed(11)
    stimulus_shape = (6, 10)
    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    stimulus_runs = [
        torch.randint(0, 2, (7, n_pixels)).double(),
        torch.randint(0, 2, (5, n_pixels)).double(),
    ]
    parameters = torch.tensor([[3.4, 5.1, 1.7, 0.42], [6.0, 4.0, 2.3, 0.60]], dtype=torch.float64)
    hrf = torch.tensor([1.0, 0.5, 0.2, 0.05], dtype=torch.float64)
    _, derivatives = _css_prediction_and_derivatives(stimulus_runs, parameters, hrf, stimulus_shape)

    delta = 1e-6
    for index in range(4):
        forward = parameters.clone()
        forward[:, index] += delta
        backward = parameters.clone()
        backward[:, index] -= delta
        numeric = (
            _css_prediction(stimulus_runs, forward, hrf, stimulus_shape)
            - _css_prediction(stimulus_runs, backward, hrf, stimulus_shape)
        ) / (2 * delta)
        torch.testing.assert_close(derivatives[:, :, index], numeric, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("stimulus_shape", [(9, 7), (4, 3)])
def test_separable_prediction_matches_dense_fields_with_repeated_frames(stimulus_shape):
    """Deduplicated frames and the separable Gaussian must reproduce the dense model.

    The refinement predicts from each run's *distinct* frames and folds the
    expansion into the HRF operator, and never forms a (voxel, pixel) field.
    Blank and repeated frames, two runs, per-voxel HRF groups, and (at 4x3) more
    distinct frames than one contraction block holds must all agree with the
    dense ``gaussian_receptive_fields`` route to float64 precision.
    """
    from fastfuncstuff.design.prf import _group_bounds, _HRFApplier, predict_prf_runwise

    generator = torch.Generator().manual_seed(3)
    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    distinct = torch.randint(0, 2, (6, n_pixels), generator=generator).double()
    distinct[0] = 0.0
    stimulus_runs = [
        distinct[torch.randint(0, 6, (19,), generator=generator)],
        torch.rand((11, n_pixels), generator=generator, dtype=torch.float64),
    ]
    parameters = torch.tensor(
        [[3.4, 5.1, 1.7, 0.42], [6.0, 4.0, 2.3, 0.60], [2.2, 2.9, 0.8, 1.0], [5.5, 1.5, 3.1, 0.25]],
        dtype=torch.float64,
    )
    library = torch.tensor([[1.0, 0.5, 0.2, 0.05], [0.3, 1.0, 0.4, 0.1]], dtype=torch.float64)
    group = torch.tensor([0, 0, 1, 1])
    applier = _HRFApplier(library, _group_bounds(group, 2))

    separable = _css_prediction(stimulus_runs, parameters, applier, stimulus_shape)
    with_derivatives, _ = _css_prediction_and_derivatives(
        stimulus_runs, parameters, applier, stimulus_shape
    )
    fields = gaussian_receptive_fields(parameters, stimulus_shape)
    dense = torch.cat(
        [
            predict_prf_runwise(
                stimulus_runs,
                fields[voxel : voxel + 1],
                parameters[voxel, 3:],
                library[group[voxel]],
            )
            for voxel in range(parameters.shape[0])
        ],
        dim=1,
    )
    torch.testing.assert_close(separable, dense, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(with_derivatives, dense, rtol=1e-12, atol=1e-14)


def test_prediction_only_path_matches_the_derivative_path():
    """The line search and the Jacobian step must optimize the identical objective."""
    data, stimulus_runs, parameters, hrf = _synthetic_css_data()
    del data
    with_derivatives, _ = _css_prediction_and_derivatives(stimulus_runs, parameters, hrf, (5, 5))
    torch.testing.assert_close(
        _css_prediction(stimulus_runs, parameters, hrf, (5, 5)), with_derivatives
    )


def test_refinement_recovers_a_batch_of_voxels_at_once():
    """Batched refinement must not depend on the batch size.

    A single-voxel batch hides shape errors in the per-voxel Levenberg damping,
    which broadcasts silently when n_voxels happens to equal the parameter count.
    """
    torch.manual_seed(7)
    stimulus_shape = (7, 7)
    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    stimulus_runs = [torch.randint(0, 2, (24, n_pixels)).float()]
    truth = torch.tensor(
        [[3.2, 4.1, 1.3, 0.5], [5.0, 2.6, 1.8, 0.5], [2.4, 5.5, 1.1, 0.5], [4.0, 4.0, 2.2, 0.5]]
    )
    hrf = torch.tensor([1.0, 0.6, 0.2])
    prediction, _ = _css_prediction_and_derivatives(stimulus_runs, truth, hrf, stimulus_shape)
    data = (prediction * torch.tensor([2.0, 3.0, 1.5, 4.0])).T.contiguous()

    seeds = truth + torch.tensor([[0.2, -0.2, 0.1, 0.0]])
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(4, dtype=torch.long),
        hrf_index=torch.zeros(4, dtype=torch.long),
        parameters=seeds,
        gain=torch.ones(4),
        correlation=torch.zeros(4),
        r2=torch.zeros(4),
    )
    fit = refine_prf_supergrid(
        data,
        stimulus_runs,
        stimulus_shape,
        grid_fit,
        hrf.unsqueeze(0),
        [0],
        voxel_chunk_size=4,
        config=PRFRefinementConfig(max_iter=40, fix_exponent=True),
    )

    assert fit.r2.min().item() > 0.999
    torch.testing.assert_close(fit.parameters[:, :3], truth[:, :3], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(fit.gain, torch.tensor([2.0, 3.0, 1.5, 4.0]), atol=5e-2, rtol=5e-2)


def test_supergrid_gain_matches_an_explicit_least_squares_solve():
    """The r * |data| / |prediction| shortcut must equal the direct OLS gain."""
    torch.manual_seed(3)
    stimulus_shape = (5, 5)
    n_pixels = 25
    stimulus_runs = [torch.randint(0, 2, (16, n_pixels)).float()]
    grid = make_analyzeprf_grid(stimulus_shape)
    hrf = torch.tensor([[1.0, 0.5, 0.1]])
    candidates = grid.parameters[[10, 40, 90]]
    fields = gaussian_receptive_fields(candidates, stimulus_shape)
    from fastfuncstuff.design.prf import predict_prf_runwise

    predictions = predict_prf_runwise(stimulus_runs, fields, candidates[:, 3], hrf[0])
    data = (predictions[:, 1] * 3.5).unsqueeze(0)

    fit = fit_prf_supergrid(
        data,
        stimulus_runs,
        stimulus_shape,
        PRFGrid(candidates),
        hrf,
        [0],
        candidate_chunk_size=2,
        voxel_chunk_size=1,
    )
    assert fit.candidate_index.item() == 1
    torch.testing.assert_close(fit.gain, torch.tensor([3.5]), atol=1e-3, rtol=1e-3)


def test_quick_mode_returns_grid_seeds_unrefined():
    """analyzePRF seedmode -2: the grid winner is passed through untouched."""
    from fastfuncstuff.design.prf import grid_seeds_as_fit

    grid_fit = PRFGridFit(
        candidate_index=torch.tensor([3, 1]),
        hrf_index=torch.tensor([0, 2]),
        parameters=torch.tensor([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]]),
        gain=torch.tensor([1.5, 2.5]),
        correlation=torch.tensor([0.8, 0.6]),
        r2=torch.tensor([0.64, 0.36]),
    )
    quick = grid_seeds_as_fit(grid_fit)

    torch.testing.assert_close(quick.parameters, grid_fit.parameters)
    torch.testing.assert_close(quick.gain, grid_fit.gain)
    torch.testing.assert_close(quick.r2, grid_fit.r2)
    assert quick.n_iters.tolist() == [0, 0]
    assert not quick.converged.any()
    assert quick.residual_ss.isnan().all()


def test_hrf_summary_recovers_a_sub_step_peak_and_flags_flat_voxels():
    """Parabolic interpolation beats the discrete argmax; flat curves score ~0 evidence."""
    from fastfuncstuff.design.prf import summarize_hrf_selection

    n_hrfs = 9
    axis = torch.arange(n_hrfs, dtype=torch.float32)
    peaked = 0.99 - 0.01 * (axis - 4.3).square()  # true peak between entries 4 and 5
    flat = torch.full((n_hrfs,), 0.5)
    r2 = torch.stack([peaked, flat])

    index, continuous, evidence = summarize_hrf_selection(r2)

    assert index.tolist()[0] == 4
    assert abs(continuous[0].item() - 4.3) < 0.05
    assert evidence[0] > 0.01
    assert evidence[1].abs() < 1e-6


def test_refine_all_hrfs_picks_the_hrf_the_grid_stage_cannot():
    """Refitting under each HRF must beat the grid's frozen choice on its own R2."""
    from fastfuncstuff.design.prf import refine_prf_all_hrfs

    torch.manual_seed(19)
    stimulus_shape = (7, 7)
    n_pixels = 49
    stimulus_runs = [torch.randint(0, 2, (40, n_pixels)).float()]
    library = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.3, 1.0], [0.2, 1.0, 0.2]])
    truth = torch.tensor([[3.1, 4.2, 1.4, 0.5], [5.0, 2.5, 1.9, 0.5]])
    true_hrf = 1
    prediction, _ = _css_prediction_and_derivatives(
        stimulus_runs, truth, library[true_hrf], stimulus_shape
    )
    data = (prediction * torch.tensor([2.0, 3.0])).T.contiguous()

    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(2, dtype=torch.long),
        # Seed with the wrong HRF on purpose: selection must be able to escape it.
        hrf_index=torch.zeros(2, dtype=torch.long),
        parameters=truth + torch.tensor([[0.2, -0.2, 0.1, 0.0]]),
        gain=torch.ones(2),
        correlation=torch.zeros(2),
        r2=torch.zeros(2),
    )
    fit, r2_map, kept = refine_prf_all_hrfs(
        data,
        stimulus_runs,
        stimulus_shape,
        grid_fit,
        library,
        [0],
        voxel_chunk_size=2,
        config=PRFRefinementConfig(max_iter=40, fix_exponent=True),
        keep_hrf_index=2,
    )

    assert r2_map.shape == (2, 3)
    assert fit.hrf_index.tolist() == [true_hrf, true_hrf]
    assert fit.r2.min().item() > 0.999
    # The returned fit must be the winning HRF's fit, not some other HRF's.
    torch.testing.assert_close(fit.r2, r2_map.amax(dim=1))
    # keep_hrf_index returns that HRF's own fit, not the winner's.
    assert kept is not None
    assert kept.hrf_index.tolist() == [2, 2]
    torch.testing.assert_close(kept.r2, r2_map[:, 2])


def _hrf_args(**overrides):
    from fastfuncstuff.cli.pyrf import create_parser

    argv = ["-input", "x", "-stimulus", "y", "-prefix", "z"]
    for flag, value in overrides.items():
        argv += [f"-{flag}"] if value is True else [f"-{flag}", str(value)]
    return create_parser().parse_args(argv)


def test_hrf_mode_canonical_returns_one_hrf():
    """-hrf canonical short-circuits the library so refinement runs once."""
    from fastfuncstuff.cli.pyrf import _build_hrf_library

    library, canonical_index = _build_hrf_library(_hrf_args(hrf="canonical"), 1.0, "cpu")

    assert library.shape[0] == 1
    assert canonical_index == 0


def test_num_hrfs_subsamples_the_library_across_its_full_range():
    """Fewer HRFs must span the same timing range, not truncate one end of it."""
    from fastfuncstuff.cli.pyrf import _build_hrf_library

    full, _ = _build_hrf_library(_hrf_args(), 1.0, "cpu")
    subset, _ = _build_hrf_library(_hrf_args(num_hrfs=5), 1.0, "cpu")

    assert subset.shape[0] == 5
    torch.testing.assert_close(subset[0], full[0])
    torch.testing.assert_close(subset[-1], full[-1])


def test_save_canonical_appends_the_canonical_as_the_last_hrf():
    """The canonical is appended, so the library's own entries are untouched."""
    from fastfuncstuff.cli.pyrf import _build_hrf_library

    library, _ = _build_hrf_library(_hrf_args(), 1.0, "cpu")
    with_canonical, canonical_index = _build_hrf_library(_hrf_args(save_canonical=True), 1.0, "cpu")

    assert canonical_index == library.shape[0]
    assert with_canonical.shape[0] == library.shape[0] + 1
    torch.testing.assert_close(with_canonical[:-1], library)


def test_pighs_mode_generates_the_requested_number_of_hrfs():
    """-hrf pighs honours -num_hrfs."""
    from fastfuncstuff.cli.pyrf import _build_hrf_library

    library, canonical_index = _build_hrf_library(_hrf_args(hrf="pighs", num_hrfs=7), 1.0, "cpu")

    assert library.shape[0] == 7
    assert canonical_index is None


def test_screen_extent_reports_position_in_degrees(tmp_path):
    """x/y/ecc_deg must be the pixel parameters scaled by degrees-per-pixel."""
    import types

    import nibabel as nib

    from fastfuncstuff.cli.pyrf import _save_results

    # One voxel, 10 pixels right of and 4 pixels above the aperture center
    # (a SMALLER row is higher in the visual field, as in analyzePRF).
    center = (1.0 + 100.0) / 2.0
    results = types.SimpleNamespace(
        parameters=torch.tensor([[center - 4.0, center + 10.0, 3.0, 0.25]]),
        gain=torch.ones(1),
        correlation=torch.ones(1),
        r2=torch.ones(1),
        hrf_index=torch.zeros(1, dtype=torch.long),
        candidate_index=torch.zeros(1, dtype=torch.long),
        residual_ss=torch.zeros(1),
        n_iters=torch.ones(1),
        converged=torch.ones(1),
    )
    loaded = types.SimpleNamespace(
        mask_flat=None, volume_shape=(1, 1, 1), affine=np.eye(4), nifti_header=None
    )
    output = tmp_path / "prf.nii.gz"

    _save_results(results, str(output), loaded, (100, 100), screen_extent=20.0)

    from fastfuncstuff.io.afni import read_brick_labels

    image = nib.load(output)
    values = dict(zip(read_brick_labels(image), image.get_fdata().ravel(), strict=True))
    degrees_per_pixel = 20.0 / 100.0
    assert values["x"] == pytest.approx(10.0 * degrees_per_pixel)
    assert values["y"] == pytest.approx(4.0 * degrees_per_pixel)
    assert values["eccentricity"] == pytest.approx(np.hypot(10.0, 4.0) * degrees_per_pixel)
    # rfsize is sigma/sqrt(n), converted with the same factor.
    assert values["rfsize"] == pytest.approx((3.0 / 0.5) * degrees_per_pixel)
    assert values["sigma"] == pytest.approx(3.0 * degrees_per_pixel)


def test_positions_are_center_relative_pixels_without_screen_extent(tmp_path):
    """Without -screen_extent, x/y are still center-relative, just in pixels."""
    import types

    import nibabel as nib

    from fastfuncstuff.cli.pyrf import _save_results
    from fastfuncstuff.io.afni import read_brick_labels

    center = (1.0 + 100.0) / 2.0
    results = types.SimpleNamespace(
        parameters=torch.tensor([[center - 4.0, center + 10.0, 3.0, 0.25]]),
        gain=torch.ones(1),
        correlation=torch.ones(1),
        r2=torch.ones(1),
        hrf_index=torch.zeros(1, dtype=torch.long),
        candidate_index=torch.zeros(1, dtype=torch.long),
        residual_ss=torch.zeros(1),
        n_iters=torch.ones(1),
        converged=torch.ones(1),
    )
    loaded = types.SimpleNamespace(
        mask_flat=None, volume_shape=(1, 1, 1), affine=np.eye(4), nifti_header=None
    )
    output = tmp_path / "prf_px.nii.gz"

    _save_results(results, str(output), loaded, (100, 100))

    image = nib.load(output)
    values = dict(zip(read_brick_labels(image), image.get_fdata().ravel(), strict=True))
    assert values["x"] == pytest.approx(10.0)
    assert values["y"] == pytest.approx(4.0)
    assert values["eccentricity"] == pytest.approx(np.hypot(10.0, 4.0))


def test_three_derivative_mode_matches_the_first_three_columns():
    """Dropping the exponent derivative must not perturb the ones that remain."""
    torch.manual_seed(11)
    stimulus_shape = (6, 6)
    stimulus_runs = [torch.rand(12, 36)]
    parameters = torch.tensor([[3.0, 4.0, 1.5, 0.6], [2.0, 2.5, 2.0, 1.0]])
    hrf = torch.tensor([1.0, 0.5, 0.1])

    full_prediction, full = _css_prediction_and_derivatives(
        stimulus_runs, parameters, hrf, stimulus_shape
    )
    linear_prediction, linear = _css_prediction_and_derivatives(
        stimulus_runs, parameters, hrf, stimulus_shape, n_derivatives=3
    )

    assert linear.shape[-1] == 3
    torch.testing.assert_close(linear_prediction, full_prediction)
    torch.testing.assert_close(linear, full[:, :, :3])


def test_multi_source_reuses_one_runs_frames_across_runs(monkeypatch):
    """A repeated identical sweep is given once and reused, not concatenated."""
    frames = torch.arange(18, dtype=torch.float32).reshape(3, 6)

    monkeypatch.setattr(
        "fastfuncstuff.cli.pyrf._load_stimulus_run",
        lambda path, downsample=0: (frames, (2, 3)),
    )

    runs, shape = _load_stimulus_sources([["one_sweep", "3"]], [3, 3, 3])

    assert shape == (2, 3)
    assert len(runs) == 3
    for run in runs:
        torch.testing.assert_close(run, frames)


def test_multi_source_rejects_a_frame_count_matching_neither_reading(monkeypatch):
    """An ambiguous-looking count is an error, not a silent guess."""
    frames = torch.zeros(4, 6)

    monkeypatch.setattr(
        "fastfuncstuff.cli.pyrf._load_stimulus_run",
        lambda path, downsample=0: (frames, (2, 3)),
    )

    with pytest.raises(ValueError, match="neither"):
        _load_stimulus_sources([["odd", "2"]], [3, 3])


def test_loro_hrf_modes_select_the_hrf_they_promise():
    """'fixed' scores the given HRF; 'grid' is free to disagree with it."""
    torch.manual_seed(23)
    stimulus_shape = (6, 6)
    n_pixels = 36
    stimulus_runs = [torch.rand(20, n_pixels), torch.rand(20, n_pixels)]
    library = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.2, 1.0]])
    truth = torch.tensor([[3.0, 3.5, 1.2, 0.5], [2.5, 4.0, 1.0, 0.5]])
    prediction, _ = _css_prediction_and_derivatives(
        stimulus_runs, truth, library[0], stimulus_shape
    )
    data = prediction.T.contiguous()
    grid = make_analyzeprf_grid(stimulus_shape, exponents=(0.5,))
    config = PRFRefinementConfig(max_iter=8, fix_exponent=True)

    forced = torch.ones(2, dtype=torch.long)  # the wrong HRF, on purpose
    fixed = fit_prf_loro(
        data,
        stimulus_runs,
        stimulus_shape,
        grid,
        library,
        [0, 20],
        candidate_chunk_size=64,
        voxel_chunk_size=2,
        refinement_config=config,
        hrf_mode="fixed",
        fixed_hrf_index=forced,
    )
    grid_mode = fit_prf_loro(
        data,
        stimulus_runs,
        stimulus_shape,
        grid,
        library,
        [0, 20],
        candidate_chunk_size=64,
        voxel_chunk_size=2,
        refinement_config=config,
    )

    # Forcing the wrong HRF must cost held-out R2 relative to letting the fold choose.
    assert (fixed.r2 < grid_mode.r2).all()


def test_loro_rejects_fixed_mode_without_an_index():
    """'fixed' has no meaning without the HRF it is supposed to fix."""
    with pytest.raises(ValueError, match="fixed_hrf_index"):
        fit_prf_loro(
            torch.zeros(1, 4),
            [torch.zeros(2, 4), torch.zeros(2, 4)],
            (2, 2),
            make_analyzeprf_grid((2, 2)),
            torch.ones(1, 2),
            [0, 2],
            hrf_mode="fixed",
        )


def test_hrf_window_slides_at_the_library_edge(monkeypatch):
    """A voxel that picked HRF 0 must still be scored on a full-width window."""
    from fastfuncstuff.design.prf import PRFRefinedFit, refine_prf_hrf_window

    tested: list[list[int]] = []

    def fake_refine(data, stimulus_runs, stimulus_shape, grid_fit, library, run_starts, **kwargs):
        tested.append(grid_fit.hrf_index.tolist())
        n = grid_fit.hrf_index.numel()
        return PRFRefinedFit(
            candidate_index=torch.zeros(n, dtype=torch.long),
            hrf_index=grid_fit.hrf_index.clone(),
            parameters=torch.zeros(n, 4),
            gain=torch.zeros(n),
            correlation=torch.zeros(n),
            # Make the middle HRF of the library the winner everywhere.
            r2=-(grid_fit.hrf_index.float() - 2.0).abs(),
            residual_ss=torch.zeros(n),
            n_iters=torch.zeros(n),
            converged=torch.zeros(n),
        )

    monkeypatch.setattr("fastfuncstuff.design.prf.refine_prf_supergrid", fake_refine)

    # Five HRFs; voxels seeded at the bottom edge, the middle, and the top edge.
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(3, dtype=torch.long),
        hrf_index=torch.tensor([0, 2, 4]),
        parameters=torch.zeros(3, 4),
        gain=torch.ones(3),
        correlation=torch.zeros(3),
        r2=torch.zeros(3),
    )
    fit = refine_prf_hrf_window(
        torch.zeros(3, 4), [torch.zeros(4, 4)], (2, 2), grid_fit, torch.ones(5, 2), [0], window=1
    )

    # Each voxel is scored on exactly three distinct HRFs, slid inside [0, 4].
    per_voxel = list(zip(*tested, strict=True))
    assert per_voxel[0] == (0, 1, 2)
    assert per_voxel[1] == (1, 2, 3)
    assert per_voxel[2] == (2, 3, 4)
    assert fit.hrf_index.tolist() == [2, 2, 2]


def test_hrf_ranked_follows_grid_fit_not_library_order(monkeypatch):
    """grid-N takes the best-scoring HRFs, however they are ordered in the library."""
    from fastfuncstuff.design.prf import PRFRefinedFit, refine_prf_hrf_ranked

    tested: list[list[int]] = []

    def fake_refine(data, stimulus_runs, stimulus_shape, grid_fit, library, run_starts, **kwargs):
        tested.append(grid_fit.hrf_index.tolist())
        n = grid_fit.hrf_index.numel()
        return PRFRefinedFit(
            candidate_index=grid_fit.candidate_index.clone(),
            hrf_index=grid_fit.hrf_index.clone(),
            parameters=grid_fit.parameters.clone(),
            gain=torch.zeros(n),
            correlation=torch.zeros(n),
            r2=grid_fit.hrf_index.float(),  # the highest index refines best
            residual_ss=torch.zeros(n),
            n_iters=torch.zeros(n),
            converged=torch.zeros(n),
        )

    monkeypatch.setattr("fastfuncstuff.design.prf.refine_prf_supergrid", fake_refine)

    # Grid scores whose ranking has nothing to do with index adjacency.
    per_hrf = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.8, 0.3],
            [0.1, 0.5, 0.9, 0.2, 0.7],
        ]
    )
    # One distinct grid candidate per (voxel, HRF), so the seed each trial got --
    # and the one the winner carries out -- is identifiable.
    candidate_per_hrf = torch.arange(10, dtype=torch.int32).reshape(2, 5)
    grid_parameters = torch.arange(10, dtype=torch.float32).reshape(10, 1).repeat(1, 4)
    grid_fit = PRFGridFit(
        candidate_index=torch.tensor([0, 7]),
        hrf_index=torch.tensor([0, 2]),
        parameters=grid_parameters[torch.tensor([0, 7])],
        gain=torch.ones(2),
        correlation=torch.zeros(2),
        r2=torch.zeros(2),
        correlation_per_hrf=per_hrf,
        candidate_per_hrf=candidate_per_hrf,
        grid_parameters=grid_parameters,
    )
    fit = refine_prf_hrf_ranked(
        torch.zeros(2, 4), [torch.zeros(4, 4)], (2, 2), grid_fit, torch.ones(5, 2), [0], n_extra=2
    )

    # n_extra=2 means three candidates: the grid winner plus the two runners-up.
    per_voxel = [sorted(scores) for scores in zip(*tested, strict=True)]
    assert per_voxel[0] == [0, 3, 4]
    assert per_voxel[1] == [1, 2, 4]
    assert fit.hrf_index.tolist() == [4, 4]
    # Each trial was seeded at its own HRF's grid candidate, and the winning
    # trial's candidate -- not the first trial's -- is the one reported.
    assert fit.candidate_index.tolist() == [4, 9]
    assert torch.equal(fit.parameters, grid_parameters[torch.tensor([4, 9])])


def test_hrf_ranked_requires_tracked_grid_scores():
    from fastfuncstuff.design.prf import refine_prf_hrf_ranked

    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(2, dtype=torch.long),
        hrf_index=torch.zeros(2, dtype=torch.long),
        parameters=torch.zeros(2, 4),
        gain=torch.ones(2),
        correlation=torch.zeros(2),
        r2=torch.zeros(2),
    )
    with pytest.raises(ValueError, match="track_per_hrf"):
        refine_prf_hrf_ranked(
            torch.zeros(2, 4), [torch.zeros(4, 4)], (2, 2), grid_fit, torch.ones(5, 2), [0]
        )


def test_supergrid_tracks_the_best_correlation_for_every_hrf():
    """correlation_per_hrf must hold each HRF's own best, and agree with the winner."""
    torch.manual_seed(11)
    stimulus_shape = (5, 5)
    stimulus_runs = [torch.rand(20, 25)]
    library = torch.tensor([[1.0, 0.4, 0.1], [0.1, 0.4, 1.0], [0.5, 1.0, 0.5]])
    grid = make_analyzeprf_grid(stimulus_shape, n_angles=8)
    data = torch.rand(6, 20)

    fit = fit_prf_supergrid(
        data, stimulus_runs, stimulus_shape, grid, library, [0], track_per_hrf=True
    )

    assert fit.correlation_per_hrf is not None
    assert fit.candidate_per_hrf is not None
    assert fit.correlation_per_hrf.shape == (6, 3)
    best_per_voxel, best_hrf = fit.correlation_per_hrf.max(dim=1)
    assert torch.allclose(best_per_voxel, fit.correlation)
    assert torch.equal(best_hrf, fit.hrf_index)

    # Each column must equal what a single-HRF grid search would have found.
    for index in range(3):
        alone = fit_prf_supergrid(
            data, stimulus_runs, stimulus_shape, grid, library[index : index + 1], [0]
        )
        assert torch.equal(alone.candidate_index, fit.candidate_per_hrf[:, index].long())
        assert torch.allclose(alone.correlation, fit.correlation_per_hrf[:, index])

    # Untracked by default, so the extra buffers are never paid for silently.
    plain = fit_prf_supergrid(data, stimulus_runs, stimulus_shape, grid, library, [0])
    assert plain.correlation_per_hrf is None
    assert plain.candidate_per_hrf is None


def test_seeds_for_hrf_uses_that_hrfs_own_grid_optimum():
    """Forcing an HRF must move the seed to where the grid liked *that* HRF."""
    torch.manual_seed(12)
    stimulus_shape = (5, 5)
    stimulus_runs = [torch.rand(20, 25)]
    library = torch.tensor([[1.0, 0.4, 0.1], [0.1, 0.4, 1.0], [0.5, 1.0, 0.5]])
    grid = make_analyzeprf_grid(stimulus_shape, n_angles=8)
    data = torch.rand(6, 20)

    fit = fit_prf_supergrid(
        data, stimulus_runs, stimulus_shape, grid, library, [0], track_per_hrf=True
    )
    forced = torch.full_like(fit.hrf_index, 1)
    seeded = fit.seeds_for_hrf(forced)

    alone = fit_prf_supergrid(data, stimulus_runs, stimulus_shape, grid, library[1:2], [0])
    assert torch.equal(seeded.candidate_index, alone.candidate_index)
    assert torch.allclose(seeded.parameters, alone.parameters)
    assert torch.allclose(seeded.correlation, alone.correlation)
    # Re-seeding at the winning HRF is a no-op.
    assert torch.allclose(fit.seeds_for_hrf(fit.hrf_index).parameters, fit.parameters)

    # Without the tracked tables the old behaviour stands: only the HRF changes.
    plain = fit_prf_supergrid(data, stimulus_runs, stimulus_shape, grid, library, [0])
    assert torch.allclose(plain.seeds_for_hrf(forced).parameters, plain.parameters)


def test_hrf_window_wider_than_the_library_falls_back_to_every_hrf():
    """A window covering the whole library is just the full search."""
    from fastfuncstuff.design.prf import refine_prf_hrf_window

    torch.manual_seed(5)
    stimulus_shape = (5, 5)
    stimulus_runs = [torch.rand(12, 25)]
    library = torch.tensor([[1.0, 0.3], [0.2, 1.0]])
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(2, dtype=torch.long),
        hrf_index=torch.zeros(2, dtype=torch.long),
        parameters=torch.tensor([[3.0, 3.0, 1.2, 0.5], [2.0, 3.5, 1.0, 0.5]]),
        gain=torch.ones(2),
        correlation=torch.zeros(2),
        r2=torch.zeros(2),
    )
    fit = refine_prf_hrf_window(
        torch.rand(2, 12),
        stimulus_runs,
        stimulus_shape,
        grid_fit,
        library,
        [0],
        window=3,
        config=PRFRefinementConfig(max_iter=5, fix_exponent=True),
    )

    assert fit.parameters.shape == (2, 4)


def test_mixed_hrf_batch_matches_refining_each_hrf_separately():
    """One full-width batch with per-voxel HRFs must equal per-HRF batches.

    This is the property the sort/scatter refactor can silently break: voxels are
    reordered by HRF before refinement, so a wrong inverse permutation would give
    every voxel a neighbour's answer while still looking plausible.
    """
    torch.manual_seed(31)
    stimulus_shape = (7, 7)
    n_pixels = 49
    stimulus_runs = [torch.rand(30, n_pixels), torch.rand(24, n_pixels)]
    run_starts = [0, 30]
    library = torch.tensor(
        [[1.0, 0.30, 0.05], [0.10, 1.00, 0.20], [0.05, 0.25, 1.00]], dtype=torch.float32
    )
    truth = torch.tensor(
        [
            [3.0, 4.0, 1.3, 0.5],
            [4.5, 2.5, 1.1, 0.5],
            [2.5, 5.0, 1.6, 0.5],
            [5.0, 4.0, 1.2, 0.5],
            [3.5, 3.0, 1.4, 0.5],
            [4.0, 5.5, 1.0, 0.5],
        ]
    )
    # Interleave the HRFs so the sort has to actually permute.
    hrf_index = torch.tensor([0, 1, 2, 0, 1, 2])
    data = torch.stack(
        [
            _css_prediction(stimulus_runs, truth[v : v + 1], library[hrf_index[v]], stimulus_shape)[
                :, 0
            ]
            for v in range(6)
        ]
    )
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(6, dtype=torch.long),
        hrf_index=hrf_index,
        parameters=truth + torch.tensor([[0.3, -0.25, 0.1, 0.0]]),
        gain=torch.ones(6),
        correlation=torch.zeros(6),
        r2=torch.zeros(6),
    )
    config = PRFRefinementConfig(max_iter=25, fix_exponent=True)
    common = dict(voxel_chunk_size=64, config=config)

    mixed = refine_prf_supergrid(
        data, stimulus_runs, stimulus_shape, grid_fit, library, run_starts, **common
    )

    # The old semantics: one batch per HRF, results scattered back by index.
    for wanted in (0, 1, 2):
        rows = torch.nonzero(hrf_index == wanted, as_tuple=False).squeeze(1)
        alone = refine_prf_supergrid(
            data[rows],
            stimulus_runs,
            stimulus_shape,
            PRFGridFit(
                candidate_index=grid_fit.candidate_index[rows],
                hrf_index=grid_fit.hrf_index[rows],
                parameters=grid_fit.parameters[rows],
                gain=grid_fit.gain[rows],
                correlation=grid_fit.correlation[rows],
                r2=grid_fit.r2[rows],
            ),
            library,
            run_starts,
            **common,
        )
        torch.testing.assert_close(mixed.parameters[rows], alone.parameters, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(mixed.r2[rows], alone.r2, atol=1e-5, rtol=1e-4)

    # And the fit must actually recover the simulated pRFs.
    torch.testing.assert_close(mixed.parameters[:, :3], truth[:, :3], atol=5e-2, rtol=5e-2)
    assert mixed.r2.min().item() > 0.999


def test_arc_angle_mode_spaces_candidates_evenly_across_rings():
    """Angles scale with ring radius, so arc spacing stops depending on eccentricity."""
    stimulus_shape = (108, 108)
    uniform = make_analyzeprf_grid(stimulus_shape, n_angles=32, exponents=(1.0,))
    arc = make_analyzeprf_grid(stimulus_shape, n_angles=32, exponents=(1.0,), angle_mode="arc")

    # Same peripheral resolution, far fewer candidates.
    assert arc.n_candidates < uniform.n_candidates / 3

    center = (1.0 + 108) / 2.0
    for grid, expect_constant in ((uniform, False), (arc, True)):
        positions = np.unique(np.round(grid.parameters[:, :2].numpy(), 4), axis=0)
        radii = np.hypot(positions[:, 0] - center, positions[:, 1] - center)
        spacings = []
        for radius in np.unique(np.round(radii, 3)):
            ring = np.isclose(radii, radius, atol=1e-3)
            count = int(ring.sum())
            if radius > 1.0 and count > 1:
                spacings.append(2 * np.pi * radius / count)
        # Uniform sampling makes arc spacing grow with radius; arc keeps it flat.
        ratio = max(spacings) / min(spacings)
        assert bool(ratio < 1.5) is expect_constant, f"spacing ratio {ratio:.2f}"

    # The outermost ring keeps the full requested angular resolution.
    positions = np.unique(np.round(arc.parameters[:, :2].numpy(), 4), axis=0)
    radii = np.hypot(positions[:, 0] - center, positions[:, 1] - center)
    assert np.isclose(radii, radii.max(), atol=1e-3).sum() == 32


def test_screening_separates_responsive_voxels_from_noise():
    """The linear screen must rank real pRF voxels above pure noise."""
    from fastfuncstuff.design.prf import screen_voxels_ridge

    torch.manual_seed(77)
    stimulus_shape = (20, 20)
    n_pixels = 400
    # Two runs of a crude sweep, so the folds are whole runs.
    stimulus_runs = []
    for _ in range(2):
        frames = torch.zeros(40, 20, 20)
        for t in range(40):
            frames[t, :, (t // 2) % 20] = 1.0
        stimulus_runs.append(frames.reshape(40, n_pixels))
    hrf = torch.tensor([0.2, 1.0, 0.6, 0.2])
    truth = torch.tensor([[8.0, 6.0, 2.0, 1.0], [12.0, 14.0, 2.5, 1.0]])
    signal = _css_prediction(stimulus_runs, truth, hrf, stimulus_shape)

    responsive = (signal.T * 3.0) + torch.randn(2, 80) * 0.2
    noise = torch.randn(6, 80)
    data = torch.cat([responsive, noise], dim=0)

    scores = screen_voxels_ridge(
        data, stimulus_runs, stimulus_shape, hrf, [0, 40], n_tiles=60, gaussians_per_tile=4
    )

    assert scores.shape == (8,)
    assert scores[:2].min() > 0.3
    assert scores[2:].max() < scores[:2].min()


def test_hashed_tiles_cover_the_aperture_without_gaps():
    """Every pixel must be reachable by some tile, or pRFs there are invisible."""
    from fastfuncstuff.design.prf import hashed_gaussian_tiles

    tiles = hashed_gaussian_tiles((30, 30), n_tiles=120, gaussians_per_tile=5, seed=3)

    assert tiles.shape == (900, 120)
    per_pixel = tiles.sum(dim=1)
    assert per_pixel.min() > 0
    # Tiles are unit-volume, so the coverage should not be wildly uneven either.
    assert (per_pixel.max() / per_pixel.min()) < 50


def test_screening_folds_use_runs_when_available():
    """Multi-run input holds out whole runs; a single run falls back to time blocks."""
    from fastfuncstuff.design.prf import _screening_folds

    by_run = _screening_folds([0, 10, 25], 40, n_blocks=4)
    assert [len(fold) for fold in by_run] == [10, 15, 15]

    by_block = _screening_folds([0], 40, n_blocks=4)
    assert [len(fold) for fold in by_block] == [10, 10, 10, 10]
    assert torch.cat(by_block).tolist() == list(range(40))


def test_screening_rejects_constant_voxels_instead_of_ranking_them_first():
    """A voxel with no variance must not score a perfect 1.0 out of 0/0."""
    from fastfuncstuff.design.prf import screen_voxels_ridge

    torch.manual_seed(9)
    stimulus_shape = (12, 12)
    stimulus_runs = [torch.rand(20, 144), torch.rand(20, 144)]
    hrf = torch.tensor([0.3, 1.0, 0.4])
    data = torch.cat([torch.randn(2, 40), torch.zeros(1, 40), torch.full((1, 40), 7.0)])

    scores = screen_voxels_ridge(
        data, stimulus_runs, stimulus_shape, hrf, [0, 20], n_tiles=40, gaussians_per_tile=3
    )

    assert torch.isneginf(scores[2:]).all()
    assert torch.isfinite(scores[:2]).all()


def test_noise_pc_count_prefers_the_fewest_within_tolerance():
    """The rule takes the knee of the curve, not its noisy argmax."""
    from fastfuncstuff.design.prf import select_noise_pc_count

    # Big jump by 2, then a negligible drift upward.
    curve = [0.30, 0.36, 0.445, 0.447, 0.448, 0.449]
    assert select_noise_pc_count(curve, tolerance=0.05) == 2
    # A stricter tolerance is allowed to keep paying for the tail.
    assert select_noise_pc_count(curve, tolerance=0.0) == 5


def test_noise_pc_count_returns_zero_when_denoising_does_not_help():
    """Components that never beat the undenoised fit must not be kept."""
    from fastfuncstuff.design.prf import select_noise_pc_count

    assert select_noise_pc_count([0.50, 0.49, 0.47, 0.44]) == 0
    assert select_noise_pc_count([0.50]) == 0


def test_appending_components_extends_every_run_block():
    """Noise PCs join the per-run nuisance, keeping the block-diagonal structure."""
    from fastfuncstuff.cli.pyrf import _append_components

    nuisance = [torch.ones(10, 3), torch.ones(8, 3)]
    components = [torch.randn(10, 5), torch.randn(8, 5)]

    unchanged = _append_components(nuisance, components, 0)
    assert [block.shape for block in unchanged] == [(10, 3), (8, 3)]

    extended = _append_components(nuisance, components, 2)
    assert [block.shape for block in extended] == [(10, 5), (8, 5)]
    torch.testing.assert_close(extended[0][:, 3:], components[0][:, :2])
    torch.testing.assert_close(extended[1][:, 3:], components[1][:, :2])


def test_grid_only_cross_validation_skips_refinement():
    """refine=False must score the seeds, which is what the noise sweep uses."""
    data, stimulus_runs, parameters, hrf = _synthetic_css_data()
    common = dict(candidate_chunk_size=1, voxel_chunk_size=1)

    seeds_only = fit_prf_loro(
        data,
        stimulus_runs,
        (5, 5),
        PRFGrid(parameters),
        hrf.unsqueeze(0),
        [0, 9],
        refine=False,
        **common,
    )
    refined = fit_prf_loro(
        data,
        stimulus_runs,
        (5, 5),
        PRFGrid(parameters),
        hrf.unsqueeze(0),
        [0, 9],
        refinement_config=PRFRefinementConfig(max_iter=10),
        **common,
    )

    # The grid here holds the true parameters, so seeds alone already predict well.
    assert seeds_only.r2.item() > 0.99
    assert refined.r2.item() > 0.99


def test_excluded_voxels_keep_their_mean_but_lose_their_fit(tmp_path):
    """Screening a voxel out blanks its fit, not the mean image."""
    import types

    import nibabel as nib

    from fastfuncstuff.cli.pyrf import _save_results
    from fastfuncstuff.io.afni import read_brick_labels

    results = types.SimpleNamespace(
        parameters=torch.tensor([[50.0, 50.0, 3.0, 0.5], [50.0, 50.0, 3.0, 0.5]]),
        gain=torch.ones(2),
        correlation=torch.ones(2),
        r2=torch.ones(2),
        hrf_index=torch.zeros(2, dtype=torch.long),
        candidate_index=torch.zeros(2, dtype=torch.long),
        residual_ss=torch.zeros(2),
        n_iters=torch.ones(2),
        converged=torch.ones(2),
    )
    loaded = types.SimpleNamespace(
        mask_flat=None, volume_shape=(2, 1, 1), affine=np.eye(4), nifti_header=None
    )
    output = tmp_path / "excluded.nii.gz"

    _save_results(
        results,
        str(output),
        loaded,
        (100, 100),
        invalid_voxels=torch.tensor([False, True]),
        mean_volume=torch.tensor([120.0, 340.0]),
    )

    image = nib.load(output)
    labels = read_brick_labels(image)
    data = image.get_fdata().reshape(2, len(labels))
    assert np.isnan(data[1, labels.index("r2")])
    assert np.isnan(data[1, labels.index("x")])
    assert data[1, labels.index("meanvol")] == pytest.approx(340.0)
    assert data[0, labels.index("meanvol")] == pytest.approx(120.0)


def _write_prf_dataset(tmp_path):
    """Two runs of a tiny volume: a responsive slab, plus non-responsive voxels."""
    import nibabel as nib

    from fastfuncstuff.design.hrf import load_canonical_hrf_basic

    rng = np.random.default_rng(0)
    shape = (12, 12, 2)
    n_voxels = int(np.prod(shape))
    n_tps = 120
    aperture = (6, 6)
    hrf = load_canonical_hrf_basic(microtime_dt=1.0, device=torch.device("cpu")).numpy()
    inputs, stimuli = [], []
    for run in range(2):
        frames = rng.integers(0, 2, (n_tps, *aperture)).astype(np.float32)
        # A blob at a fixed aperture location, so the responsive voxels have a
        # position the grid can actually find.
        signal = np.convolve(frames[:, 1:3, 1:3].mean(axis=(1, 2)), hrf)[:n_tps]
        nuisance = rng.standard_normal(n_tps).astype(np.float32)
        series = np.empty((n_voxels, n_tps), dtype=np.float32)
        for voxel in range(n_voxels):
            noise = 0.05 * rng.standard_normal(n_tps).astype(np.float32)
            if voxel < 24:  # the fit mask, below
                series[voxel] = 100.0 + 10.0 * signal + noise
            else:
                series[voxel] = 100.0 + 5.0 * nuisance + noise
        image = nib.Nifti1Image(series.reshape(*shape, n_tps), np.eye(4))
        image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
        path = tmp_path / f"run{run}.nii.gz"
        nib.save(image, path)
        inputs.append(str(path))
        stimulus = tmp_path / f"stim{run}.nii.gz"
        nib.save(nib.Nifti1Image(frames.transpose(1, 2, 0)[:, :, None, :], np.eye(4)), stimulus)
        stimuli.append(str(stimulus))

    fit_mask = np.zeros(n_voxels, dtype=np.float32)
    fit_mask[:24] = 1.0
    mask_path = tmp_path / "fitmask.nii.gz"
    nib.save(nib.Nifti1Image(fit_mask.reshape(shape), np.eye(4)), mask_path)
    return inputs, stimuli, str(mask_path), shape


def test_noise_pool_mask_reaches_outside_the_fit_mask(tmp_path):
    """-noise_pool_mask fits only inside -mask but pools noise from the whole volume."""
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.afni import read_brick_labels

    inputs, stimuli, mask_path, shape = _write_prf_dataset(tmp_path)
    prefix = str(tmp_path / "out")
    assert (
        main(
            [
                "-input",
                *inputs,
                "-stimulus",
                *stimuli,
                "-mask",
                mask_path,
                "-prefix",
                prefix,
                "-denoise",
                "-noise_pool_mask",
                "all",
                "-max_pcs",
                "2",
                "-save_denoise",
                "-quick",
                "-screen_tiles",
                "40",
                "-grid_angles",
                "4",
                "-grid_sigma_steps",
                "1",
                "-grid_ecc_steps",
                "2",
                "-device",
                "cpu",
                "-quiet",
            ]
        )
        == 0
    )

    image = nib.load(f"{prefix}.nii.gz")
    labels = read_brick_labels(image)
    r2 = image.get_fdata().reshape(-1, len(labels))[:, labels.index("r2")]
    # Everything was loaded (the pool is the whole volume) but only the fit mask
    # was fit; the rest comes back NaN rather than as a fit of the wrong voxels.
    assert np.isfinite(r2[:24]).all()
    assert np.isnan(r2[24:]).all()
    # ... and the pool it denoised with is drawn from voxels the fit never saw.
    pool = nib.load(f"{prefix}_noisepool.nii.gz").get_fdata().reshape(-1)
    assert pool[24:].sum() > 0
    assert pool[:24].sum() == 0


def test_average_ranks_share_the_rank_within_a_tie_group():
    """Bound-clamped parameters tie constantly; ordinal ranks would invent an order."""
    from fastfuncstuff.stats.reliability import average_ranks

    ranks = average_ranks(torch.tensor([10.0, 20.0, 20.0, 30.0]))

    torch.testing.assert_close(ranks, torch.tensor([1.0, 2.5, 2.5, 4.0]))


def test_circular_correlation_survives_the_angle_wrap():
    """A constant rotation is perfect circular agreement and terrible linear agreement."""
    from fastfuncstuff.stats.reliability import circular_correlation, spearman_correlation

    # Concentrated, so the circular mean is well defined; the rotation then
    # carries part of the sample across the 2*pi -> 0 seam.
    angles = torch.linspace(0.0, 2.0, 50)
    rotated = (angles + 5.5).remainder(2 * torch.pi)
    assert rotated.max() > 6.0 and rotated.min() < 1.0

    assert circular_correlation(angles, rotated) == pytest.approx(1.0, abs=1e-6)
    # The same data through the linear coefficient the other parameters use.
    assert abs(spearman_correlation(angles, rotated)) < 0.6


def test_noise_ceiling_recovers_a_known_signal_fraction():
    """With y = s + e, the correlation between repeats is var(s) / var(y) -- the R2 cap."""
    from fastfuncstuff.stats.reliability import split_half_noise_ceiling

    generator = torch.Generator().manual_seed(3)
    n_timepoints = 4000
    signal = torch.randn(1, n_timepoints, generator=generator)
    first = signal + torch.randn(1, n_timepoints, generator=generator)
    second = signal + torch.randn(1, n_timepoints, generator=generator)
    data = torch.cat([first, second], dim=1)

    ceiling = split_half_noise_ceiling(data, [[0, 1]], [0, n_timepoints], 2 * n_timepoints)

    # var(s) = var(e) = 1, so the reproducible fraction is 0.5.
    assert ceiling.item() == pytest.approx(0.5, abs=0.03)


def test_noise_ceiling_is_nan_without_repeated_designs():
    """No repeats means no ceiling, which must be distinguishable from a zero ceiling."""
    from fastfuncstuff.stats.reliability import split_half_noise_ceiling

    data = torch.randn(3, 20)

    ceiling = split_half_noise_ceiling(data, [], [0, 10], 20)

    assert torch.isnan(ceiling).all()


def test_identical_design_labels_group_only_bit_identical_apertures():
    """The ceiling needs identical designs, which is stricter than 'same stimulus family'."""
    from fastfuncstuff.stats.reliability import identical_design_groups, identical_design_labels

    bars = torch.randint(0, 2, (7, 9), dtype=torch.float32)
    wedges = torch.randint(0, 2, (7, 9), dtype=torch.float32)
    runs = [bars, bars.clone(), wedges, bars]

    assert identical_design_labels(runs) == [0, 0, 1, 0]
    assert identical_design_groups(runs) == [[0, 1, 3]]


def test_balanced_halves_cover_every_group_and_share_no_runs():
    """Both halves must be complete stimulus sets, and disjoint, or neither number means anything."""
    from fastfuncstuff.design.prf import balanced_group_halves

    labels = [1, 1, 1, 2, 2, 2]
    draws = balanced_group_halves(labels, n_draws=3, seed=0)

    assert draws
    for first, second in draws:
        assert not set(first) & set(second)
        for half in (first, second):
            assert {labels[index] for index in half} == {1, 2}


def test_balanced_halves_rotate_which_run_sits_out():
    """Odd groups leave one run out per draw; a fixed choice would bias the estimate."""
    from fastfuncstuff.design.prf import balanced_group_halves

    draws = balanced_group_halves([1, 1, 1, 2, 2, 2], n_draws=4, seed=0)

    used = {tuple(sorted([*first, *second])) for first, second in draws}
    assert len(used) > 1


def test_balanced_halves_skip_groups_with_a_single_run():
    """A lone run cannot be in both halves, so its stimulus leaves the analysis."""
    from fastfuncstuff.design.prf import balanced_group_halves

    draws = balanced_group_halves([0, 0, 1], n_draws=2, seed=0)

    assert draws
    for first, second in draws:
        assert 2 not in first and 2 not in second
        assert len(first) == len(second) == 1


def test_balanced_halves_report_nothing_when_no_group_repeats():
    """Every group a singleton means no valid split exists; the caller must be told."""
    from fastfuncstuff.design.prf import balanced_group_halves

    assert balanced_group_halves([0, 1, 2], n_draws=3, seed=0) == []


def _four_run_css_data():
    """Four runs of one noiseless CSS voxel, so folds can hold out more than one run."""
    torch.manual_seed(11)
    stimulus_runs = [torch.randint(0, 2, (12, 25), dtype=torch.float32) for _ in range(4)]
    parameters = torch.tensor([[2.8, 3.2, 1.1, 0.65]], dtype=torch.float32)
    hrf = torch.tensor([1.0, 0.4, 0.1], dtype=torch.float32)
    prediction, _ = _css_prediction_and_derivatives(stimulus_runs, parameters, hrf, (5, 5))
    return (prediction.T * 2.5).contiguous(), stimulus_runs, parameters, hrf


def test_folds_can_hold_out_several_runs_at_once():
    """Half-splits hold out multiple runs; the held-out score must still be exact."""
    from fastfuncstuff.design.prf import fit_prf_folds

    data, stimulus_runs, parameters, hrf = _four_run_css_data()
    run_starts = [0, 12, 24, 36]

    results, fold_fits = fit_prf_folds(
        data,
        stimulus_runs,
        (5, 5),
        PRFGrid(parameters),
        hrf.unsqueeze(0),
        run_starts,
        [([0, 1], [2, 3]), ([2, 3], [0, 1])],
        candidate_chunk_size=1,
        voxel_chunk_size=1,
        refinement_config=PRFRefinementConfig(max_iter=10),
        return_fits=True,
    )

    assert results.r2.item() > 0.999
    assert len(fold_fits) == 2


def test_folds_reject_a_run_that_is_both_trained_on_and_held_out():
    """Overlapping folds would silently report a training score as held-out."""
    from fastfuncstuff.design.prf import fit_prf_folds

    data, stimulus_runs, parameters, hrf = _four_run_css_data()

    with pytest.raises(ValueError, match="cannot both train on and hold out"):
        fit_prf_folds(
            data,
            stimulus_runs,
            (5, 5),
            PRFGrid(parameters),
            hrf.unsqueeze(0),
            [0, 12, 24, 36],
            [([0, 1], [1, 2])],
        )


def test_fold_fits_are_only_returned_when_asked():
    """LORO keeps n_folds x n_voxels parameter sets out of memory unless they are wanted."""
    from fastfuncstuff.design.prf import fit_prf_folds, loro_folds

    data, stimulus_runs, parameters, hrf = _four_run_css_data()

    _, fold_fits = fit_prf_folds(
        data,
        stimulus_runs,
        (5, 5),
        PRFGrid(parameters),
        hrf.unsqueeze(0),
        [0, 12, 24, 36],
        loro_folds(4),
        candidate_chunk_size=1,
        voxel_chunk_size=1,
        refinement_config=PRFRefinementConfig(max_iter=5),
    )

    assert fold_fits == []


def test_reliability_curve_stops_when_too_few_voxels_survive():
    """A correlation over a handful of voxels is noise; the sweep must not report it."""
    from fastfuncstuff.cli.pyrf import _reliability_curve

    generator = torch.Generator().manual_seed(5)
    n_voxels = 400
    parameters = torch.rand(n_voxels, 4, generator=generator) * 5 + 1
    pairs = [
        (
            {name: values for name, values in _named_maps(parameters).items()},
            {name: values for name, values in _named_maps(parameters).items()},
        )
    ]
    # R2 decreasing across voxels, so each threshold keeps a known count.
    r2 = torch.linspace(1.0, 0.0, n_voxels)

    curve = _reliability_curve(pairs, r2, step=0.1, min_voxels=100)

    assert curve
    assert all(row["n_voxels"] >= 100 for row in curve)
    # Identical halves agree perfectly, which is the sanity check on the plumbing.
    assert curve[0]["rfsize"] == pytest.approx(1.0, abs=1e-6)


def _named_maps(parameters):
    from fastfuncstuff.design.prf import prf_parameter_maps

    return prf_parameter_maps(parameters, (10, 10), 1.0)


def test_vertical_sign_matches_analyzeprf():
    """Up is toward row 0, as in analyzePRF -- the sign is silently reversible."""
    stimulus_shape = (100, 100)
    center = (1.0 + 100.0) / 2.0
    rows = torch.tensor([center - 4.0, center + 4.0, center])
    parameters = torch.stack(
        [rows, torch.full_like(rows, center + 10.0), torch.ones_like(rows), torch.ones_like(rows)],
        dim=1,
    )

    maps = prf_parameter_maps(parameters, stimulus_shape, 1.0)

    # analyzePRF: ang = atan2((1+res(1))/2 - R, C - (1+res(2))/2).
    expected = np.rad2deg(np.arctan2(center - rows.numpy(), 10.0)) % 360.0
    np.testing.assert_allclose(maps["y"].numpy(), [4.0, -4.0, 0.0])
    np.testing.assert_allclose(maps["angle"].numpy(), expected, atol=1e-5)
    # A pRF above the horizontal meridian is in the first quadrant, not the fourth.
    assert 0.0 < maps["angle"][0] < 90.0


def _write_grouped_prf_dataset(tmp_path):
    """Four runs in two stimulus groups: runs 0,1 share one aperture, runs 2,3 another."""
    import nibabel as nib

    from fastfuncstuff.design.hrf import load_canonical_hrf_basic

    rng = np.random.default_rng(7)
    shape = (12, 12, 1)
    n_voxels = int(np.prod(shape))
    n_tps = 80
    aperture = (6, 6)
    hrf = load_canonical_hrf_basic(microtime_dt=1.0, device=torch.device("cpu")).numpy()

    inputs, stimuli = [], []
    group_frames = [rng.integers(0, 2, (n_tps, *aperture)).astype(np.float32) for _ in range(2)]
    for run in range(4):
        frames = group_frames[run // 2]
        series = np.empty((n_voxels, n_tps), dtype=np.float32)
        for voxel in range(n_voxels):
            # Reliability is an across-voxel correlation, so the voxels have to
            # differ: each one gets its own aperture position to recover.
            row, column = (voxel // 12) % 5, (voxel % 12) % 5
            signal = np.convolve(
                frames[:, row : row + 2, column : column + 2].mean(axis=(1, 2)), hrf
            )[:n_tps]
            noise = 0.3 * rng.standard_normal(n_tps).astype(np.float32)
            series[voxel] = 100.0 + 10.0 * signal + noise
        image = nib.Nifti1Image(series.reshape(*shape, n_tps), np.eye(4))
        image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
        path = tmp_path / f"grun{run}.nii.gz"
        nib.save(image, path)
        inputs.append(str(path))

        stimulus = tmp_path / f"gstim{run}.nii.gz"
        nib.save(nib.Nifti1Image(frames.transpose(1, 2, 0)[:, :, None, :], np.eye(4)), stimulus)
        stimuli.append(str(stimulus))
    return inputs, stimuli


def test_xval_halves_reports_reliability_and_a_noise_ceiling(tmp_path):
    """The halves scheme yields held-out R2 and split-half reliability from one set of fits."""
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.afni import read_brick_labels

    inputs, stimuli = _write_grouped_prf_dataset(tmp_path)
    prefix = str(tmp_path / "halves")
    assert (
        main(
            [
                "-input",
                *inputs,
                "-stimulus",
                *stimuli,
                "-prefix",
                prefix,
                "-xval",
                "halves",
                "-stim_groups",
                "bars",
                "bars",
                "wedges",
                "wedges",
                "-hrf",
                "canonical",
                "-grid_angles",
                "4",
                "-maxiter",
                "8",
                "-device",
                "cpu",
                "-quiet",
            ]
        )
        == 0
    )

    image = nib.load(f"{prefix}.nii.gz")
    labels = read_brick_labels(image)
    # The ceiling exists because runs within a group share a bit-identical aperture.
    assert "noise_ceiling" in labels
    assert "xval_r2_normalized" in labels
    values = image.get_fdata().reshape(-1, len(labels))
    ceiling = values[:, labels.index("noise_ceiling")]
    assert np.isfinite(ceiling).all() and (ceiling > 0).all()

    curve = (tmp_path / "halves_reliability.tsv").read_text().strip().splitlines()
    header = curve[0].split("\t")
    assert "rfsize" in header and "rfsize_sb" in header
    assert len(curve) > 1
    # Position is the control: it is reliable in any working pRF fit.
    first_row = dict(zip(header, curve[1].split("\t"), strict=True))
    assert float(first_row["eccentricity"]) > 0.5
    assert (tmp_path / "halves_reliability.png").exists()
    assert (tmp_path / "halves_reliability.nii.gz").exists()


def test_halves_refuses_a_design_with_no_repeated_group(tmp_path):
    """Without a group of two or more runs there are no valid disjoint halves."""
    from fastfuncstuff.cli.pyrf import main

    inputs, stimuli = _write_grouped_prf_dataset(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "-input",
                *inputs[:2],
                "-stimulus",
                *stimuli[:2],
                "-prefix",
                str(tmp_path / "nogroups"),
                "-xval",
                "halves",
                "-stim_groups",
                "a",
                "b",
                "-hrf",
                "canonical",
                "-grid_angles",
                "4",
                "-device",
                "cpu",
                "-quiet",
            ]
        )


def test_tr_boxcar_kernel_reproduces_fine_grained_convolution():
    """The per-TR kernel must be the TR-INTEGRATED HRF, not the sampled impulse response.

    A pRF design has no onset matrix: the aperture is one frame per TR and the
    drive is one value per TR, so the HRF has to absorb the width of a sample.
    Writing S = TR/dt fine samples per TR and g[j] = sum_u h[jS - u], a stimulus
    on for k TRs gives

        sum_s g[t-s] = sum_s sum_u h[(t-s)S - u] = sum_{j<kS} h[tS - j]

    because (sS + u) covers 0..kS-1 exactly once. So it equals the fine-grained
    convolution EXACTLY, for any k -- the boxcar is a sample-width correction and
    is not duplicated as the stimulus lengthens. Pinned here because reverting to
    stim_duration=0 would look harmless and silently shift every pRF position by
    half a TR along the stimulus sweep.
    """
    from fastfuncstuff.design.hrf import _load_hrf_from_file

    impulse = _load_hrf_from_file("getcanonicalbasic.tsv")
    samples_per_tr = 10  # TR = 1.0 s against the 0.1 s file resolution
    tr_kernel = np.convolve(impulse, np.ones(samples_per_tr))[::samples_per_tr]

    for n_tr_on in (1, 3, 8):
        fine = np.zeros(len(impulse) + samples_per_tr * n_tr_on)
        fine[: samples_per_tr * n_tr_on] = 1.0
        truth = np.convolve(fine, impulse)[::samples_per_tr]

        drive = np.zeros(len(truth))
        drive[:n_tr_on] = 1.0
        predicted = np.convolve(drive, tr_kernel)[: len(truth)]

        np.testing.assert_allclose(predicted, truth, atol=1e-12)


def test_prf_hrfs_are_tr_integrated(monkeypatch):
    """Every HRF ffs_pyrf builds carries the one-TR boxcar, for all -hrf modes."""
    import types

    from fastfuncstuff.cli.pyrf import _build_hrf_library
    from fastfuncstuff.design.hrf import load_canonical_hrf_basic

    tr = 1.0
    device = torch.device("cpu")
    args = types.SimpleNamespace(
        hrf_mode="canonical",
        hrf_duration=32.0,
        hrf_library=None,
        num_hrfs=None,
        save_canonical=False,
    )
    canonical, _ = _build_hrf_library(args, tr, device)
    expected = load_canonical_hrf_basic(
        microtime_dt=tr, hrf_duration=32.0, stim_duration=tr, device=device
    )
    torch.testing.assert_close(canonical[0], expected)

    # The impulse-sampled kernel is what this used to be; it must NOT match.
    impulse = load_canonical_hrf_basic(microtime_dt=tr, hrf_duration=32.0, device=device)
    assert not torch.allclose(canonical[0], impulse, atol=1e-3)

    from fastfuncstuff.design.hrf import get_hrf_library, load_canonical_hrf_library

    args.hrf_mode = "library"
    library, _ = _build_hrf_library(args, tr, device)
    torch.testing.assert_close(
        library,
        load_canonical_hrf_library(
            microtime_dt=tr, hrf_duration=32.0, stim_duration=tr, device=device
        ),
    )

    args.hrf_mode = "pighs"
    args.seed = 0
    pighs, _ = _build_hrf_library(args, tr, device)
    torch.testing.assert_close(
        pighs,
        get_hrf_library(
            mode="pighs",
            microtime_dt=tr,
            hrf_duration=32.0,
            stim_duration=tr,
            n_hrfs=20,
            seed=0,
            device=device,
        ),
    )


def test_pighs_library_is_reproducible_under_a_seed():
    """The PIGHS parameters are Latin-hypercube samples; unseeded they differ per call."""
    from fastfuncstuff.design.hrf import get_hrf_library

    device = torch.device("cpu")
    kwargs = dict(mode="pighs", microtime_dt=1.0, hrf_duration=32.0, n_hrfs=8, device=device)
    torch.testing.assert_close(get_hrf_library(seed=7, **kwargs), get_hrf_library(seed=7, **kwargs))
    assert not torch.allclose(get_hrf_library(seed=7, **kwargs), get_hrf_library(seed=8, **kwargs))


def test_prf_canonical_matches_the_analyzeprf_reference_hrf():
    """ffs_pyrf's canonical must agree with analyzePRF's getcanonicalhrf(tr,tr).

    The underlying vector is bit-identical (both come from Kay's canonical, ours
    via GLMdenoise), so the only room for disagreement is the boxcar and the
    resample. The residual is linear vs pchip interpolation onto the TR grid.
    """
    import types

    from fastfuncstuff.cli.pyrf import _build_hrf_library

    reference_path = Path(__file__).parent / "data" / "cvn_hrf_tr1.1D"
    if not reference_path.exists():
        pytest.skip("analyzePRF reference HRF fixture not present")
    reference = np.loadtxt(reference_path)

    args = types.SimpleNamespace(
        hrf_mode="canonical",
        hrf_duration=32.0,
        hrf_library=None,
        num_hrfs=None,
        save_canonical=False,
    )
    canonical, _ = _build_hrf_library(args, 1.0, torch.device("cpu"))
    ours = canonical[0].numpy()

    n = min(len(ours), len(reference))
    assert np.abs(ours[:n] - reference[:n]).max() < 5e-3
    assert int(np.argmax(ours)) == int(np.argmax(reference))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hoisted_prf_drive_matches_per_hrf_predictions(dtype):
    from fastfuncstuff.design.prf import _convolve_prf_drives, predict_prf_runwise

    g = torch.Generator().manual_seed(981)
    stimuli = [torch.rand(n, 12, generator=g, dtype=dtype) for n in (17, 11)]
    fields = torch.rand(5, 12, generator=g, dtype=dtype)
    exponents = torch.linspace(0.2, 1.0, 5, dtype=dtype)
    hrfs = torch.rand(3, 7, generator=g, dtype=dtype)
    drives = [(run @ fields.T).clamp_min(0).pow(exponents) for run in stimuli]
    for hrf in hrfs:
        old = predict_prf_runwise(stimuli, fields, exponents, hrf)
        new = _convolve_prf_drives(drives, hrf)
        torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_supergrid_drive_hoist_matches_per_hrf_fit(monkeypatch):
    import fastfuncstuff.design.prf as prf

    g = torch.Generator().manual_seed(193)
    stimuli = [torch.rand(n, 9, generator=g) for n in (17, 13)]
    parameters = torch.tensor([[0.0, 0.0, 1.0, 0.5], [0.5, -0.5, 1.5, 0.8], [1.0, 1.0, 0.8, 1.0]])
    grid = PRFGrid(parameters)
    library = torch.rand(3, 5, generator=g)
    data = torch.rand(4, 30, generator=g)
    kwargs = dict(
        candidate_chunk_size=3,
        voxel_chunk_size=2,
        track_per_hrf=True,
        nuisance_per_run=[torch.ones(n, 1) for n in (17, 13)],
    )
    new = prf.fit_prf_supergrid(data, stimuli, (3, 3), grid, library, [0, 17], **kwargs)
    old = prf._convolve_prf_drives
    fields = prf.gaussian_receptive_fields(parameters, (3, 3))

    def unhoisted(drives, hrf):
        return old([(run @ fields.T).clamp_min(0).pow(parameters[:, 3]) for run in stimuli], hrf)

    monkeypatch.setattr(prf, "_convolve_prf_drives", unhoisted)
    reference = prf.fit_prf_supergrid(data, stimuli, (3, 3), grid, library, [0, 17], **kwargs)
    for name in ("candidate_index", "hrf_index", "gain", "correlation", "correlation_per_hrf"):
        torch.testing.assert_close(getattr(new, name), getattr(reference, name), rtol=0, atol=0)


def test_short_source_is_padded_with_blank_frames(monkeypatch, capsys):
    """A stimulus program that stops before the scan does gets blank tail frames."""
    frames = torch.ones(3, 6)

    monkeypatch.setattr(
        "fastfuncstuff.cli.pyrf._load_stimulus_run",
        lambda path, downsample=0: (frames, (2, 3)),
    )

    runs, shape = _load_stimulus_sources(["short"], [5])

    assert shape == (2, 3)
    assert runs[0].shape == (5, 6)
    torch.testing.assert_close(runs[0][:3], frames)
    assert runs[0][3:].abs().max() == 0
    assert "padding the last 2 TR(s)" in capsys.readouterr().out


def test_short_reused_source_is_padded_once_per_run(monkeypatch):
    """A reused sweep shorter than its runs pads to one run's length, not the concatenation."""
    frames = torch.ones(3, 6)

    monkeypatch.setattr(
        "fastfuncstuff.cli.pyrf._load_stimulus_run",
        lambda path, downsample=0: (frames, (2, 3)),
    )

    runs, _ = _load_stimulus_sources([["one_sweep", "2"]], [4, 4])

    assert [run.shape for run in runs] == [(4, 6), (4, 6)]


@pytest.mark.parametrize(
    ("hrf_mode", "expected"),
    [("library", "grid+1"), ("canonical", "grid+1"), ("pighs", "grid-2")],
)
def test_hrf_select_defaults_to_a_window_matched_to_the_library(hrf_mode, expected, tmp_path):
    """The default refines a narrow window, ranked for pighs because it is unordered."""
    from fastfuncstuff.cli.pyrf import _default_hrf_select, create_parser

    parsed = create_parser().parse_args(
        ["-input", str(tmp_path / "run.nii.gz"), "-hrf", hrf_mode, "-prefix", "out.nii.gz"]
    )

    # None at parse time so an explicit -hrf_select is distinguishable from the
    # default; main() resolves it against the library actually requested.
    assert parsed.hrf_select is None
    assert _default_hrf_select(parsed.hrf_mode) == expected


@pytest.mark.gpu
def test_frame_composed_hrf_operator_is_reproducible():
    """The unique-frame HRF operator must be bit-identical every time it is built.

    It sums Toeplitz columns over repeated frames. Built with an on-device
    index_add_, that sum is an atomic scatter on MPS/CUDA whose order changes
    between builds, and the last-bit difference sent whole refinements down
    different paths run to run.
    """
    from fastfuncstuff.design.prf import _compress_aperture_runs, _HRFApplier
    from fastfuncstuff.utils import get_device

    device = get_device()
    if device.type == "cpu":
        pytest.skip("needs a GPU backend")
    generator = torch.Generator().manual_seed(0)
    distinct = torch.rand((9, 16), generator=generator)
    stimulus = distinct[torch.randint(0, 9, (300,), generator=generator)].to(device)
    (run,) = _compress_aperture_runs([stimulus], (4, 4))
    hrf = torch.rand(30, generator=generator).to(device)
    builds = [_HRFApplier(hrf)._matrices(run.n_timepoints, run) for _ in range(10)]
    assert all(torch.equal(builds[0], other) for other in builds[1:])


def test_shape_bucket_padding_leaves_the_refinement_unchanged(monkeypatch):
    """Padded working sets (the MPS path) must give the unpadded answer exactly.

    On MPS the refinement pads every subset -- Jacobian refreshes, line-search
    trials, the shrinking live set, each HRF group's convolution window -- to a
    shape bucket, so finished voxels ride along as inactive rows. A leak from
    one of those rows (a write-back, a convergence flag, a neighbour's HRF
    columns) would look plausible and only ever happen on a Mac. Forcing the
    padding on the CPU makes it testable; float64 keeps rounding from batch
    shape out of the comparison.
    """
    import fastfuncstuff.design.prf as prf_module

    generator = torch.Generator().manual_seed(8)
    stimulus_shape = (6, 6)
    distinct = torch.randint(0, 2, (7, 36), generator=generator).double()
    stimulus_runs = [
        distinct[torch.randint(0, 7, (28,), generator=generator)],
        distinct[torch.randint(0, 7, (22,), generator=generator)],
    ]
    run_starts = [0, 28]
    library = torch.tensor(
        [[1.0, 0.30, 0.05], [0.10, 1.00, 0.20], [0.05, 0.25, 1.00]], dtype=torch.float64
    )
    n_voxels = 30
    truth = torch.stack(
        [
            torch.rand(n_voxels, generator=generator, dtype=torch.float64) * 3 + 2,
            torch.rand(n_voxels, generator=generator, dtype=torch.float64) * 3 + 2,
            torch.rand(n_voxels, generator=generator, dtype=torch.float64) + 0.8,
            torch.rand(n_voxels, generator=generator, dtype=torch.float64) * 0.5 + 0.3,
        ],
        dim=1,
    )
    hrf_index = torch.arange(n_voxels) % 3
    clean = torch.stack(
        [
            _css_prediction(stimulus_runs, truth[v : v + 1], library[hrf_index[v]], stimulus_shape)[
                :, 0
            ]
            for v in range(n_voxels)
        ]
    )
    # Graded noise spreads convergence over many iterations, so the live set
    # shrinks repeatedly and the line search leaves different voxels pending.
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    data = clean + noise * torch.linspace(0.0, 0.1, n_voxels, dtype=torch.float64).view(-1, 1)
    # Seeds stay well-posed: an exponent pushed to its bound flattens the
    # prediction, every trial then ties the current fit, and which side of
    # `trial_ss < current_ss` a tie lands on is rounding -- a fair difference in
    # the iteration count, but not the leak this test is looking for.
    jitter = torch.tensor([0.3, 0.3, 0.1, 0.05], dtype=torch.float64)
    seeds = truth + torch.randn(truth.shape, generator=generator, dtype=torch.float64) * jitter
    grid_fit = PRFGridFit(
        candidate_index=torch.zeros(n_voxels, dtype=torch.long),
        hrf_index=hrf_index,
        parameters=seeds,
        gain=torch.ones(n_voxels, dtype=torch.float64),
        correlation=torch.zeros(n_voxels, dtype=torch.float64),
        r2=torch.zeros(n_voxels, dtype=torch.float64),
    )
    config = PRFRefinementConfig(max_iter=30, stagewise_exponent=True)

    def fit():
        return refine_prf_supergrid(
            data, stimulus_runs, stimulus_shape, grid_fit, library, run_starts, config=config
        )

    unpadded = fit()
    monkeypatch.setattr(
        prf_module,
        "shape_bucket",
        lambda n, device, limit=None: n if limit is None else min(limit, n + 5),
    )
    padded = fit()

    assert unpadded.n_iters.float().std() > 0, "convergence must be staggered to exercise padding"
    # A leak would be O(1); what remains is float64 rounding from batch shape.
    torch.testing.assert_close(padded.parameters, unpadded.parameters, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(padded.r2, unpadded.r2, rtol=1e-9, atol=1e-9)
    assert torch.equal(padded.n_iters, unpadded.n_iters)
    assert torch.equal(padded.converged, unpadded.converged)


def test_amplitude_psc_is_the_response_peak_over_the_voxel_mean(tmp_path):
    """amplitude is the fitted response height; amplitude_psc divides by the mean.

    gain alone is not an amplitude: the drive carries the Gaussian's unit-mass
    normalisation and then the exponent, so two voxels with the same response
    height but different sigma report different gains.
    """
    import types

    import nibabel as nib

    from fastfuncstuff.cli.pyrf import _save_results
    from fastfuncstuff.io.afni import read_brick_labels

    results = types.SimpleNamespace(
        parameters=torch.tensor([[50.0, 50.0, 3.0, 0.5], [50.0, 50.0, 3.0, 0.5]]),
        gain=torch.ones(2),
        correlation=torch.ones(2),
        r2=torch.ones(2),
        hrf_index=torch.zeros(2, dtype=torch.long),
        candidate_index=torch.zeros(2, dtype=torch.long),
        residual_ss=torch.zeros(2),
        n_iters=torch.ones(2),
        converged=torch.ones(2),
    )
    loaded = types.SimpleNamespace(
        mask_flat=None, volume_shape=(2, 1, 1), affine=np.eye(4), nifti_header=None
    )
    output = tmp_path / "amp.nii.gz"
    _save_results(
        results,
        str(output),
        loaded,
        (100, 100),
        mean_volume=torch.tensor([100.0, 200.0]),
        response_peak=torch.tensor([2.0, 2.0]),
    )

    labels = read_brick_labels(nib.load(output))
    data = nib.load(output).get_fdata().reshape(2, len(labels))
    assert data[0, labels.index("amplitude")] == pytest.approx(2.0)
    # Same peak, half the percentage, because the second voxel's mean is twice as big.
    assert data[0, labels.index("amplitude_psc")] == pytest.approx(2.0)
    assert data[1, labels.index("amplitude_psc")] == pytest.approx(1.0)


def test_percent_signal_change_refuses_a_non_positive_mean():
    from fastfuncstuff.design.prf import prf_percent_signal_change

    psc = prf_percent_signal_change(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([50.0, 0.0, -3.0]))
    assert psc[0] == pytest.approx(2.0)
    assert torch.isnan(psc[1:]).all()


def test_response_peak_is_gain_times_its_own_predictor_peak():
    """And it is taken chunked, so it never holds a (time, voxels) prediction."""
    from fastfuncstuff.design.prf import _css_prediction, prf_response_peak

    torch.manual_seed(0)
    stimulus_runs = [torch.rand(24, 64), torch.rand(24, 64)]
    parameters = torch.tensor([[4.0, 4.0, 2.0, 0.6], [5.0, 3.0, 1.5, 0.4], [4.5, 4.5, 3.0, 1.0]])
    hrf_library = torch.rand(2, 8)
    hrf_index = torch.tensor([0, 1, 0])
    gain = torch.tensor([2.0, 0.5, 3.0])

    peak = prf_response_peak(
        stimulus_runs, parameters, hrf_index, gain, hrf_library, (8, 8), voxel_chunk_size=1
    )
    for voxel in range(3):
        expected = (
            _css_prediction(
                stimulus_runs,
                parameters[voxel : voxel + 1],
                hrf_library[hrf_index[voxel]],
                (8, 8),
            ).amax()
            * gain[voxel]
        )
        assert peak[voxel] == pytest.approx(float(expected), rel=1e-6)
    # One chunk or three, same answer.
    whole = prf_response_peak(
        stimulus_runs, parameters, hrf_index, gain, hrf_library, (8, 8), voxel_chunk_size=99
    )
    torch.testing.assert_close(peak, whole)


def test_amplitude_psc_recovers_a_simulated_response_size(tmp_path):
    """End to end: the amplitude map has to read the response that was simulated.

    The fixture's responsive voxels are ``100 + 10 * conv(frames, hrf)``, whose
    peak sits 10.3% above the voxel mean (a touch inflated: it is the max of a
    noisy series). This is the test that catches measuring the peak from zero
    instead of from the fitted baseline -- the aperture is on for about half of
    every run, so the predictor carries a large mean that the per-run
    polynomials absorb, and the unprojected peak read 28.3% here.

    It also pins the two scaling paths together: ``-do_scale`` divides by a run
    mean of 100 up front, ``-dont_scale`` divides by the voxel's own mean at the
    end, and amplitude_psc has to come out the same either way.
    """
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.afni import read_brick_labels

    inputs, stimuli, mask_path, _shape = _write_prf_dataset(tmp_path)
    recovered = {}
    for flags, tag in (([], "scaled"), (["-dont_scale"], "unscaled")):
        output = tmp_path / f"amp_{tag}.nii.gz"
        assert (
            main(
                [
                    "-input",
                    *inputs,
                    "-stimulus",
                    *stimuli,
                    "-mask",
                    mask_path,
                    "-prefix",
                    str(output),
                    "-device",
                    "cpu",
                    "-verb",
                    "0",
                    *flags,
                ]
            )
            == 0
        )
        image = nib.load(output)
        labels = read_brick_labels(image)
        assert "amplitude" in labels and "amplitude_psc" in labels
        data = image.get_fdata().reshape(-1, len(labels))
        responsive = slice(0, 24)
        recovered[tag] = float(np.nanmedian(data[responsive, labels.index("amplitude_psc")]))
        amplitude = float(np.nanmedian(data[responsive, labels.index("amplitude")]))
        mean_volume = float(np.nanmedian(data[responsive, labels.index("meanvol")]))
        # amplitude is in the units the fit saw; the percentage is it over the mean.
        assert recovered[tag] == pytest.approx(100.0 * amplitude / mean_volume, rel=0.02)

    # Measured 9.59% (scaled) and 9.58% (unscaled) against a simulated 10.3%.
    assert recovered["scaled"] == pytest.approx(recovered["unscaled"], rel=0.02)
    assert 7.0 < recovered["scaled"] < 13.0, recovered


def test_response_peak_measures_above_the_projected_baseline():
    """Projecting the predictor is what makes the peak an excursion, not a level."""
    from fastfuncstuff.design.prf import prf_response_peak

    torch.manual_seed(0)
    # Stimulus on for most of the run, so the predictor carries a big mean.
    stimulus_runs = [torch.rand(30, 64).clamp_min(0.4)]
    parameters = torch.tensor([[4.0, 4.0, 2.0, 1.0]])
    hrf_library = torch.rand(1, 6)
    kwargs = dict(
        hrf_index=torch.zeros(1, dtype=torch.long),
        gain=torch.ones(1),
        hrf_library=hrf_library,
        stimulus_shape=(8, 8),
        voxel_chunk_size=8,
    )
    from_zero = prf_response_peak(stimulus_runs, parameters, **kwargs)
    # An intercept is the baseline the per-run polynomials always carry.
    intercept = [torch.ones(30, 1) / math.sqrt(30.0)]
    above_baseline = prf_response_peak(
        stimulus_runs, parameters, **kwargs, nuisance_per_run=intercept, run_starts=[0]
    )
    assert float(above_baseline) < float(from_zero)


def test_response_peak_requires_run_starts_with_nuisance():
    from fastfuncstuff.design.prf import prf_response_peak

    with pytest.raises(ValueError, match="go together"):
        prf_response_peak(
            [torch.rand(8, 64)],
            torch.tensor([[4.0, 4.0, 2.0, 1.0]]),
            torch.zeros(1, dtype=torch.long),
            torch.ones(1),
            torch.rand(1, 4),
            (8, 8),
            voxel_chunk_size=4,
            nuisance_per_run=[torch.ones(8, 1)],
        )


class TestPerTypePrediction:
    """Splitting the aperture by stimulus type has to be an exact decomposition."""

    def _runs(self):
        torch.manual_seed(0)
        return (
            [torch.rand(30, 64), torch.rand(24, 64)],
            torch.tensor([[4.0, 4.0, 2.0, 0.6], [5.0, 3.0, 1.5, 0.4]]),
            torch.rand(6),
            (8, 8),
        )

    def test_one_type_everywhere_is_the_plain_prediction(self):
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.ones(run.shape[0], dtype=torch.long) for run in runs]
        by_type = css_prediction_by_type(runs, params, hrf, shape, labels, n_types=1)
        torch.testing.assert_close(by_type[..., 0], _css_prediction(runs, params, hrf, shape))

    def test_types_sum_back_to_the_combined_prediction(self):
        """The property the whole design rests on: the split is additive.

        Partition the timepoints between two types, give both the same pRF, and
        the two predictors must add up to the one the combined model makes. If
        this fails the per-type fit is not a reparametrisation of the old one
        and no comparison between them means anything.

        Measured at 1.2e-7 (one float32 ULP) -- the streams reduce in a
        different order than the single pass, so this is not bit-equality.
        """
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.where(torch.arange(run.shape[0]) % 2 == 0, 1, 2) for run in runs]
        by_type = css_prediction_by_type(runs, params, hrf, shape, labels, n_types=2)
        torch.testing.assert_close(
            by_type.sum(-1), _css_prediction(runs, params, hrf, shape), rtol=1e-6, atol=1e-6
        )

    def test_label_zero_is_dropped_from_every_stream(self):
        """0 means nothing was shown, so it drives no type."""
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.where(torch.arange(run.shape[0]) % 3 == 0, 0, 1) for run in runs]
        blanked = css_prediction_by_type(runs, params, hrf, shape, labels, n_types=1)
        masked = [
            run * (label == 1).to(run.dtype).unsqueeze(1)
            for run, label in zip(runs, labels, strict=True)
        ]
        torch.testing.assert_close(blanked[..., 0], _css_prediction(masked, params, hrf, shape))

    def test_free_parameters_match_tied_when_the_sets_are_identical(self):
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.where(torch.arange(run.shape[0]) % 2 == 0, 1, 2) for run in runs]
        tied = css_prediction_by_type(runs, params, hrf, shape, labels, n_types=2)
        per_type = params.unsqueeze(1).expand(-1, 2, -1).contiguous()
        free = css_prediction_by_type(
            runs, per_type, hrf, shape, labels, n_types=2, per_type_parameters=True
        )
        torch.testing.assert_close(free, tied)

    def test_a_per_type_parameter_moves_only_its_own_type(self):
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.where(torch.arange(run.shape[0]) % 2 == 0, 1, 2) for run in runs]
        per_type = params.unsqueeze(1).expand(-1, 2, -1).contiguous()
        before = css_prediction_by_type(
            runs, per_type, hrf, shape, labels, n_types=2, per_type_parameters=True
        )
        moved = per_type.clone()
        moved[:, 1, 0] = 2.0  # shift type 2's pRF row
        after = css_prediction_by_type(
            runs, moved, hrf, shape, labels, n_types=2, per_type_parameters=True
        )
        assert not torch.allclose(after[..., 1], before[..., 1])
        torch.testing.assert_close(after[..., 0], before[..., 0])

    def test_label_length_must_match_the_run(self):
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        labels = [torch.ones(5, dtype=torch.long), torch.ones(24, dtype=torch.long)]
        with pytest.raises(ValueError, match="timepoints"):
            css_prediction_by_type(runs, params, hrf, shape, labels, n_types=1)

    def test_one_label_vector_per_run(self):
        from fastfuncstuff.design.prf import css_prediction_by_type

        runs, params, hrf, shape = self._runs()
        with pytest.raises(ValueError, match="one label vector per run"):
            css_prediction_by_type(
                runs, params, hrf, shape, [torch.ones(30, dtype=torch.long)], n_types=1
            )

    def test_streams_share_one_frame_table(self):
        """The shared table is what lets the tied drive be computed once."""
        from fastfuncstuff.design.prf import _compress_aperture_runs, split_aperture_by_type

        runs, _params, _hrf, shape = self._runs()
        run = _compress_aperture_runs(runs[:1], shape)[0]
        labels = torch.where(torch.arange(run.n_timepoints) % 2 == 0, 1, 2)
        streams = split_aperture_by_type(run, labels, 3)
        assert len({stream.frames.data_ptr() for stream in streams}) == 1
        assert all(stream.n_frames == run.n_frames + 1 for stream in streams)
        # Each stream points its off-type timepoints at the appended blank frame.
        assert int((streams[2].inverse == run.n_frames).sum()) == run.n_timepoints


class TestPerTypeGains:
    """The per-type gain solve: exact NNLS, not a clamped unconstrained solve."""

    def test_one_type_reproduces_the_single_gain_path(self):
        from fastfuncstuff.design.prf import (
            _variable_projection_sse,
            variable_projection_sse_by_type,
        )

        torch.manual_seed(0)
        prediction = torch.randn(40, 200).abs()
        data = torch.randn(40, 200)
        _, by_type = variable_projection_sse_by_type(data, prediction.unsqueeze(-1))
        torch.testing.assert_close(
            by_type, _variable_projection_sse(data, prediction), rtol=1e-5, atol=1e-4
        )

    @pytest.mark.parametrize(("n_types", "correlation"), [(2, 0.0), (3, 0.9), (4, 0.97)])
    def test_matches_scipy_nnls(self, n_types, correlation):
        """Against an independent NNLS, with the constraint genuinely active.

        The correlated cases are the point: 195 of 200 voxels at corr=0.9 have
        at least one gain pinned at zero, so this exercises the active-set
        search rather than the interior solution. Measured agreement 1.1e-7 on
        the gains and 2.2e-7 relative on the SSE.
        """
        from scipy.optimize import nnls

        from fastfuncstuff.design.prf import variable_projection_sse_by_type

        torch.manual_seed(0)
        n_time, n_voxels = 40, 60
        predictions = torch.randn(n_time, n_voxels, n_types) + correlation * torch.randn(
            n_time, 1, 1
        )
        data = torch.randn(n_time, n_voxels)
        gains, sse = variable_projection_sse_by_type(data, predictions)

        n_constrained = 0
        for voxel in range(n_voxels):
            design = predictions[:, voxel, :].numpy().astype(np.float64)
            target = data[:, voxel].numpy().astype(np.float64)
            reference, _ = nnls(design, target)
            n_constrained += int((reference == 0).any())
            np.testing.assert_allclose(gains[voxel].numpy(), reference, rtol=1e-4, atol=1e-5)
            expected_sse = float(((target - design @ reference) ** 2).sum())
            assert float(sse[voxel]) == pytest.approx(expected_sse, rel=1e-4, abs=1e-4)
        if correlation:
            assert n_constrained > n_voxels // 2, "constraint never became active"

    def test_beats_clamping_an_unconstrained_solve(self):
        """The shortcut is a different answer, not a cheaper one.

        Dropping a negative coefficient changes the optimum of the ones that
        remain, so re-solving without it recovers error that clamping leaves
        behind. Measured: worse on 190 of 200 voxels, by up to 8.7 in SSE.
        """
        from fastfuncstuff.design.prf import variable_projection_sse_by_type

        torch.manual_seed(0)
        predictions = torch.randn(40, 200, 3) + 0.97 * torch.randn(40, 1, 1)
        data = torch.randn(40, 200)
        _, sse = variable_projection_sse_by_type(data, predictions)

        gram = torch.einsum("tvi,tvj->vij", predictions, predictions)
        cross = torch.einsum("tvi,tv->vi", predictions, data)
        clamped = torch.linalg.solve(gram, cross.unsqueeze(-1)).squeeze(-1).clamp_min(0)
        clamped_sse = ((data - torch.einsum("tvi,vi->tv", predictions, clamped)) ** 2).sum(0)
        # Never better than the exact solution, and usually worse.
        assert bool((clamped_sse >= sse - 1e-3).all())
        assert int((clamped_sse > sse + 1e-3).sum()) > 100

    def test_gains_are_never_negative_and_zero_is_available(self):
        from fastfuncstuff.design.prf import variable_projection_sse_by_type

        torch.manual_seed(1)
        predictions = torch.rand(30, 50, 2) + 0.5
        # Data that runs against both predictors: the best non-negative fit is 0.
        data = -(predictions.sum(-1))
        gains, sse = variable_projection_sse_by_type(data, predictions)
        assert bool((gains >= 0).all())
        torch.testing.assert_close(gains, torch.zeros_like(gains))
        torch.testing.assert_close(sse, data.square().sum(dim=0))

    def test_refuses_more_types_than_the_enumeration_handles(self):
        from fastfuncstuff.design.prf import MAX_STIMULUS_TYPES, nonnegative_gains

        too_many = MAX_STIMULUS_TYPES + 1
        with pytest.raises(ValueError, match="at most"):
            nonnegative_gains(torch.eye(too_many).expand(3, -1, -1), torch.ones(3, too_many))

    def test_collinear_predictors_do_not_throw(self):
        """A type that never appeared leaves an all-zero predictor column."""
        from fastfuncstuff.design.prf import variable_projection_sse_by_type

        torch.manual_seed(2)
        predictions = torch.rand(20, 10, 3)
        predictions[..., 2] = 0.0  # type 3 never shown
        data = torch.rand(20, 10)
        gains, sse = variable_projection_sse_by_type(data, predictions)
        assert torch.isfinite(gains).all() and torch.isfinite(sse).all()
        assert float(gains[:, 2].abs().max()) < 1e-3


class TestFitPerTypeGains:
    """The tied model: one pRF, a gain per stimulus type."""

    def _problem(self, truth):
        """Voxels whose per-type gains are *truth*, on a shared pRF."""
        from fastfuncstuff.design.prf import css_prediction_by_type
        from fastfuncstuff.glm.core import construct_polynomial_matrix

        torch.manual_seed(0)
        n_time, shape = 120, (8, 8)
        runs = [torch.rand(n_time, 64), torch.rand(n_time, 64)]
        # Blocked: 20 frames of one type, then 20 of the other.
        labels = [torch.where(torch.arange(n_time) % 40 < 20, 1, 2) for _ in runs]
        hrf = torch.rand(1, 6)
        hrf = hrf / hrf.sum()
        n_voxels = truth.shape[0]
        parameters = torch.tensor([[4.0, 4.0, 2.0, 1.0]]).expand(n_voxels, -1).contiguous()
        predictors = css_prediction_by_type(runs, parameters, hrf[0], shape, labels, 2)
        clean = torch.einsum("tvi,vi->tv", predictors, truth)
        data = (100.0 + clean + 0.05 * torch.randn(2 * n_time, n_voxels)).T
        nuisance = [
            construct_polynomial_matrix(n_time, 1, device=torch.device("cpu")) for _ in runs
        ]
        return dict(
            data=data,
            stimulus_runs=runs,
            parameters=parameters,
            hrf_index=torch.zeros(n_voxels, dtype=torch.long),
            hrf_library=hrf,
            stimulus_shape=shape,
            type_labels=labels,
            n_types=2,
            run_starts=[0, n_time],
            nuisance_per_run=nuisance,
            voxel_chunk_size=2,
        )

    def test_recovers_a_type_preference_and_flags_it_in_delta_r2(self):
        """A voxel that prefers one type, one that does not, and a dead one.

        Measured: gains 9.96/1.89 against a simulated 10/2 with delta_r2 0.94;
        6.048/6.053 against 6/6 with delta_r2 0.000; zeros for no response.
        The flat voxel is the one that matters -- where the brain does not
        distinguish the types, the per-type gains have to come out the same, or
        every map built on this would show differences that are not there.
        """
        from fastfuncstuff.design.prf import fit_prf_type_gains

        truth = torch.tensor([[10.0, 2.0], [6.0, 6.0], [0.0, 0.0]])
        fit = fit_prf_type_gains(**self._problem(truth))

        assert fit.gains[0, 0] == pytest.approx(10.0, rel=0.05)
        assert fit.gains[0, 1] == pytest.approx(2.0, rel=0.10)
        # Doesn't care: the two gains agree to well under a percent.
        assert fit.gains[1, 0] == pytest.approx(float(fit.gains[1, 1]), rel=0.01)
        assert float(fit.gains[2].abs().max()) < 0.1

        # Separating the types only buys R2 where there is a difference to find.
        assert fit.delta_r2[0] > 0.5
        assert abs(float(fit.delta_r2[1])) < 0.01
        torch.testing.assert_close(fit.delta_r2, fit.r2 - fit.single_r2)

    def test_single_r2_is_the_one_gain_null_on_the_same_prf(self):
        """delta_r2 must isolate the type split, not an HRF or pRF difference."""
        from fastfuncstuff.design.prf import fit_prf_type_gains

        truth = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
        fit = fit_prf_type_gains(**self._problem(truth))
        # Equal gains: the extra parameter is free to be useless, and is.
        torch.testing.assert_close(fit.r2, fit.single_r2, rtol=1e-3, atol=1e-3)
        assert bool((fit.r2 >= fit.single_r2 - 1e-4).all()), "more freedom cannot fit worse"

    def test_amplitude_is_the_gain_times_its_own_projected_peak(self):
        from fastfuncstuff.design.prf import fit_prf_type_gains

        truth = torch.tensor([[10.0, 2.0], [6.0, 6.0]])
        problem = self._problem(truth)
        fit = fit_prf_type_gains(**problem)
        assert fit.amplitude.shape == (2, 2)
        # Ordering follows the gains within a voxel: a bigger gain on the same
        # pRF is a bigger response.
        assert (fit.amplitude[0, 0] > fit.amplitude[0, 1]) == (fit.gains[0, 0] > fit.gains[0, 1])
        assert bool((fit.amplitude >= 0).all())

    def test_rejects_a_voxel_count_mismatch_and_a_bad_type_count(self):
        from fastfuncstuff.design.prf import MAX_STIMULUS_TYPES, fit_prf_type_gains

        problem = self._problem(torch.tensor([[1.0, 1.0]]))
        with pytest.raises(ValueError, match="voxels"):
            fit_prf_type_gains(**{**problem, "data": problem["data"][:0]})
        with pytest.raises(ValueError, match="n_types must be"):
            fit_prf_type_gains(**{**problem, "n_types": MAX_STIMULUS_TYPES + 1})
