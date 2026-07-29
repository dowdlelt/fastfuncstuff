"""
Tests for denoise/heldout.py — fully held-out prediction for ffs_denoisatorial.

The point of the held-out path is that nothing about the held-out runs feeds
the fit, and nothing about the fit touches the held-out data. These tests pin
both halves: betas come from the input runs (denoised), and the held-out runs
are only polynomial-projected before being predicted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.denoise.heldout import heldout_prediction_r2

TR = 2.0
RUN_LEN = 80


def _boxcar_design(n_runs: int, seed: int = 0) -> torch.Tensor:
    """(n_runs*RUN_LEN, 1) design: one condition, blocks every 20 TRs."""
    rng = np.random.default_rng(seed)
    col = np.zeros(n_runs * RUN_LEN, dtype=np.float32)
    for r in range(n_runs):
        for onset in range(5, RUN_LEN - 10, 20):
            col[r * RUN_LEN + onset : r * RUN_LEN + onset + 5] = 1.0
    col += rng.normal(0, 0.01, col.shape).astype(np.float32)  # break exact collinearity
    return torch.from_numpy(col[:, None])


def _polys(n_runs: int) -> list[torch.Tensor]:
    """Intercept + linear trend per run (the minimum legal nuisance block)."""
    t = torch.linspace(-1, 1, RUN_LEN)
    block = torch.stack([torch.ones(RUN_LEN), t], dim=1)
    return [block.clone() for _ in range(n_runs)]


def _zero_pcs(n_runs: int, k: int = 3) -> list[torch.Tensor]:
    return [torch.zeros(RUN_LEN, k) for _ in range(n_runs)]


def test_heldout_r2_high_for_signal_voxels_zero_for_noise():
    """Betas fit on the training runs must predict the held-out runs."""
    torch.manual_seed(0)
    n_train, n_test = 3, 2
    train_design = _boxcar_design(n_train, seed=1)
    test_design = _boxcar_design(n_test, seed=2)

    # Voxel 0: strong task response. Voxel 1: noise only.
    amp = torch.tensor([[5.0], [0.0]])
    train_data = amp @ train_design.T + torch.randn(2, n_train * RUN_LEN) * 0.5
    test_data = amp @ test_design.T + torch.randn(2, n_test * RUN_LEN) * 0.5

    r2 = heldout_prediction_r2(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=_zero_pcs(n_train),
        train_selections=[() for _ in range(n_train)],
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        device=torch.device("cpu"),
        verbose=False,
    )

    assert r2.shape == (2,)
    assert r2[0] > 0.5, f"task voxel should be predicted, got {r2[0]:.3f}"
    assert r2[1] < 0.1, f"noise voxel should not be, got {r2[1]:.3f}"


def test_denoising_the_fit_improves_the_held_out_prediction():
    """Denoising acts on the fit: cleaner betas must predict clean data better.

    The training runs carry an artifact that is partly correlated with the
    design, so leaving it in the model biases the task beta. The held-out runs
    are clean and are never touched by any PC — only the beta changes.
    """
    torch.manual_seed(0)
    n_train, n_test = 3, 2
    train_design = _boxcar_design(n_train, seed=1)
    test_design = _boxcar_design(n_test, seed=2)

    amp = torch.tensor([[4.0]])
    train_data = amp @ train_design.T + torch.randn(1, n_train * RUN_LEN) * 0.3
    test_data = amp @ test_design.T + torch.randn(1, n_test * RUN_LEN) * 0.3

    # Artifact = design + noise, so it inflates the beta if left unmodelled.
    train_pcs = []
    for r in range(n_train):
        sl = slice(r * RUN_LEN, (r + 1) * RUN_LEN)
        artifact = train_design[sl, 0] + 0.5 * torch.randn(RUN_LEN)
        artifact = (artifact - artifact.mean()) / artifact.std()
        train_data[:, sl] += 8.0 * artifact
        pcs = torch.zeros(RUN_LEN, 3)
        pcs[:, 0] = artifact
        train_pcs.append(pcs)

    common = dict(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=train_pcs,
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        device=torch.device("cpu"),
        verbose=False,
    )

    r2_raw = heldout_prediction_r2(train_selections=[() for _ in range(n_train)], **common)
    r2_denoised = heldout_prediction_r2(train_selections=[(0,)] * n_train, **common)

    assert r2_denoised[0] > r2_raw[0] + 0.2, (
        f"denoising the fit should lift held-out R²: {r2_raw[0]:.3f} -> {r2_denoised[0]:.3f}"
    )


def test_held_out_data_is_never_pc_cleaned():
    """R² must be scored against the raw (polynomial-projected) held-out data.

    Changing which PCs the *fit* removes cannot change the held-out target, so
    the total variance of the scored signal is identical across selections —
    only the residual moves. A regression that started projecting PCs out of
    the held-out runs would break this by construction (it can only raise R²).
    """
    torch.manual_seed(1)
    n_train, n_test = 3, 2
    train_design = _boxcar_design(n_train, seed=1)
    test_design = _boxcar_design(n_test, seed=2)

    # A voxel with no task response at all: any positive R² would mean the
    # held-out data had structure removed from it.
    train_data = torch.randn(1, n_train * RUN_LEN)
    test_data = torch.randn(1, n_test * RUN_LEN)
    train_pcs = [torch.randn(RUN_LEN, 3) for _ in range(n_train)]

    common = dict(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=train_pcs,
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        device=torch.device("cpu"),
        verbose=False,
    )

    r2_none = heldout_prediction_r2(train_selections=[() for _ in range(n_train)], **common)
    r2_all = heldout_prediction_r2(train_selections=[(0, 1, 2)] * n_train, **common)

    assert abs(r2_none[0]) < 0.2 and abs(r2_all[0]) < 0.2, (
        f"a pure-noise voxel cannot be predicted: {r2_none[0]:.3f}, {r2_all[0]:.3f}"
    )


def test_mismatched_voxel_counts_raise():
    n_train, n_test = 2, 1
    with pytest.raises(ValueError, match="same voxels"):
        heldout_prediction_r2(
            train_data=torch.randn(4, n_train * RUN_LEN),
            train_run_starts=[r * RUN_LEN for r in range(n_train)],
            train_nuisance_per_run=_polys(n_train),
            train_pcs_per_run=_zero_pcs(n_train),
            train_selections=[() for _ in range(n_train)],
            test_data=torch.randn(3, n_test * RUN_LEN),
            test_run_starts=[0],
            test_nuisance_per_run=_polys(n_test),
            train_design=_boxcar_design(n_train),
            test_design=_boxcar_design(n_test),
            device=torch.device("cpu"),
            verbose=False,
        )


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


def _variance_only_case(n_train: int, n_test: int, n_voxels: int, seed: int):
    """Training runs carry a design-orthogonal artifact; held-out runs are clean.

    Orthogonalising the artifact against the design is the point: it inflates
    Var(β̂) without biasing E[β̂], which is the regime where an all-runs
    held-out R² goes nearly flat and only the learning curve can see anything.
    """
    torch.manual_seed(seed)
    train_design = _boxcar_design(n_train, seed=1)
    test_design = _boxcar_design(n_test, seed=2)

    amp = torch.linspace(3.0, 6.0, n_voxels)[:, None]
    train_data = amp @ train_design.T + torch.randn(n_voxels, n_train * RUN_LEN) * 0.3
    test_data = amp @ test_design.T + torch.randn(n_voxels, n_test * RUN_LEN) * 0.3

    train_pcs = []
    for r in range(n_train):
        sl = slice(r * RUN_LEN, (r + 1) * RUN_LEN)
        artifact = torch.randn(RUN_LEN)
        d = train_design[sl, 0]
        artifact = artifact - d * (artifact @ d) / (d @ d)  # no bias, pure variance
        artifact = (artifact - artifact.mean()) / artifact.std()
        train_data[:, sl] += torch.linspace(4.0, 9.0, n_voxels)[:, None] * artifact[None, :]
        pcs = torch.zeros(RUN_LEN, 3)
        pcs[:, 0] = artifact
        pcs[:, 1:] = torch.randn(RUN_LEN, 2)
        train_pcs.append(pcs)

    return train_design, test_design, train_data, test_data, train_pcs


def test_learning_curve_separates_most_at_small_k():
    """The premise of the curve: the denoising gap shrinks as k grows.

    With a design-orthogonal artifact the betas are unbiased either way, so the
    only thing denoising buys is lower Var(β̂) — a term that enters held-out
    SS_res as Var(β̂)·||x_test||² and therefore shrinks like 1/k. If this
    ordering ever inverts, the curve is not measuring what it claims to.
    """
    from fastfuncstuff.denoise.heldout import heldout_learning_curve

    n_train, n_test, n_voxels = 6, 2, 12
    train_design, test_design, train_data, test_data, train_pcs = _variance_only_case(
        n_train, n_test, n_voxels, seed=3
    )

    curve = heldout_learning_curve(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=train_pcs,
        arms={
            "initial": [() for _ in range(n_train)],
            "denoised": [(0,)] * n_train,
        },
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        subset_sizes=[1, 6],
        max_subsets=6,
        device=torch.device("cpu"),
        verbose=False,
    )

    curves = curve["curves"]
    delta = (curves["denoised"] - curves["initial"]).median(dim=1).values
    assert delta[0] > 0, f"denoising should help at k=1, got {delta[0]:+.4f}"
    assert delta[0] > delta[1], (
        f"gap must shrink as k grows: k=1 {delta[0]:+.4f} vs k=6 {delta[1]:+.4f}"
    )


def test_learning_curve_arms_share_subsets():
    """Arms must be paired: identical selections in two arms give identical curves.

    If the subsets were redrawn per arm, two arms with the same PC selection
    would differ by the sampling noise of the draw.
    """
    from fastfuncstuff.denoise.heldout import heldout_learning_curve

    n_train, n_test, n_voxels = 5, 2, 6
    train_design, test_design, train_data, test_data, train_pcs = _variance_only_case(
        n_train, n_test, n_voxels, seed=4
    )

    curve = heldout_learning_curve(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=train_pcs,
        arms={"a": [(0,)] * n_train, "b": [(0,)] * n_train},
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        subset_sizes=[2],
        max_subsets=3,
        device=torch.device("cpu"),
        verbose=False,
    )
    torch.testing.assert_close(curve["curves"]["a"], curve["curves"]["b"])


def test_learning_curve_full_k_matches_the_single_shot_prediction():
    """At k = N there is one subset, so the curve must reproduce heldout_prediction_r2."""
    from fastfuncstuff.denoise.heldout import heldout_learning_curve

    n_train, n_test, n_voxels = 4, 2, 8
    train_design, test_design, train_data, test_data, train_pcs = _variance_only_case(
        n_train, n_test, n_voxels, seed=5
    )
    sel = [(0,), (0, 1), (), (1,)]
    common = dict(
        train_data=train_data,
        train_run_starts=[r * RUN_LEN for r in range(n_train)],
        train_nuisance_per_run=_polys(n_train),
        train_pcs_per_run=train_pcs,
        test_data=test_data,
        test_run_starts=[r * RUN_LEN for r in range(n_test)],
        test_nuisance_per_run=_polys(n_test),
        train_design=train_design,
        test_design=test_design,
        device=torch.device("cpu"),
        verbose=False,
    )

    direct = heldout_prediction_r2(train_selections=sel, **common)
    curve = heldout_learning_curve(
        arms={"denoised": sel}, subset_sizes=[n_train], max_subsets=1, **common
    )
    torch.testing.assert_close(curve["curves"]["denoised"][0], direct, rtol=1e-5, atol=1e-6)


def test_learning_curve_rejects_out_of_range_k():
    from fastfuncstuff.denoise.heldout import heldout_learning_curve

    n_train, n_test, n_voxels = 3, 1, 4
    train_design, test_design, train_data, test_data, train_pcs = _variance_only_case(
        n_train, n_test, n_voxels, seed=6
    )
    with pytest.raises(ValueError, match="outside 1..3"):
        heldout_learning_curve(
            train_data=train_data,
            train_run_starts=[r * RUN_LEN for r in range(n_train)],
            train_nuisance_per_run=_polys(n_train),
            train_pcs_per_run=train_pcs,
            arms={"initial": [() for _ in range(n_train)]},
            test_data=test_data,
            test_run_starts=[0],
            test_nuisance_per_run=_polys(n_test),
            train_design=train_design,
            test_design=test_design,
            subset_sizes=[0, 4],
            device=torch.device("cpu"),
            verbose=False,
        )


def test_learning_curve_requires_a_design():
    from fastfuncstuff.denoise.heldout import heldout_learning_curve

    n_train, n_test, n_voxels = 3, 1, 4
    _, _, train_data, test_data, train_pcs = _variance_only_case(n_train, n_test, n_voxels, seed=7)
    with pytest.raises(ValueError, match="provide either"):
        heldout_learning_curve(
            train_data=train_data,
            train_run_starts=[r * RUN_LEN for r in range(n_train)],
            train_nuisance_per_run=_polys(n_train),
            train_pcs_per_run=train_pcs,
            arms={"initial": [() for _ in range(n_train)]},
            test_data=test_data,
            test_run_starts=[0],
            test_nuisance_per_run=_polys(n_test),
            device=torch.device("cpu"),
            verbose=False,
        )
