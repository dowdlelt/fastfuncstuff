"""Tests for the CSS pRF primitives and the ffs_pyrf input helpers."""

from __future__ import annotations

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

    # One voxel, 10 pixels right of and 4 pixels above the aperture center.
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
    assert values["y"] == pytest.approx(-4.0 * degrees_per_pixel)
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
    assert values["y"] == pytest.approx(-4.0)
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
