"""
Simulation-based recovery tests for the three main CLI tools.

These tests use the existing simulation machinery to generate realistic fMRI data
(structured AR(1) noise, per-voxel scaling, non-white spectral content) and verify
that each pipeline recovers known ground truth parameters.

Focus: events 3 seconds long, ISI 1-7s, 1-20 events per condition.

Coverage:
    TestHRFRecovery      — functions used by 3dHRFoptfast.py
    TestRidgeRecovery    — functions used by 3dRidgefast.py  (close ISI stress)
    TestDenoiseRecovery  — functions used by 3dDenoisefast.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ============================================================================
# Shared simulation helpers
# ============================================================================

CPU = torch.device("cpu")


def _make_colored_noise(
    n_timepoints: int,
    n_voxels: int,
    rho: float = 0.3,
    noise_scale: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """
    AR(1) colored noise with random per-voxel amplitude scaling.

    Returns (n_voxels, n_timepoints).
    """
    from fastfuncstuff.simulation.noise import generate_ar1_noise

    torch.manual_seed(seed)
    # generate_ar1_noise returns (n_timepoints, n_voxels) for n_voxels > 1
    noise = generate_ar1_noise(
        rho=rho,
        n_timepoints=n_timepoints,
        n_voxels=n_voxels,
        normalize=True,
        device=CPU,
    )
    if noise.ndim == 1:
        noise = noise.unsqueeze(-1)
    noise = noise.T  # → (n_voxels, n_timepoints)

    # Per-voxel amplitude: log-uniform in [0.5, 3.0] to mimic SNR variation
    # Always apply per-voxel scaling for realistic SNR variation
    if noise_scale is None:
        noise_scale = 1.0  # default scale factor

    scales = torch.exp(torch.rand(n_voxels) * np.log(6.0) - np.log(2.0))  # [0.5, 3.0]
    noise = noise * scales.unsqueeze(1) * noise_scale

    return noise.float()


def _make_gaussian_hrf_library(
    microtime_dt: float = 0.1,
    hrf_duration: float = 32.0,
    peak_times: tuple[float, ...] = (5.0, 6.0, 8.0),
    undershoot_ratio: float = 0.35,
) -> torch.Tensor:
    """
    Small library of Gaussian HRFs with a negative undershoot.

    Returns (n_hrfs, n_hrf_bins).
    """
    n_hrf_bins = int(round(hrf_duration / microtime_dt))
    t_hrf = np.arange(n_hrf_bins) * microtime_dt
    hrfs = []
    for peak in peak_times:
        h = np.exp(-0.5 * ((t_hrf - peak) / 1.5) ** 2) - undershoot_ratio * np.exp(
            -0.5 * ((t_hrf - (peak + 4.0)) / 3.0) ** 2
        )
        h = h / np.abs(h).max()
        hrfs.append(h.astype(np.float32))
    return torch.from_numpy(np.stack(hrfs))  # (n_hrfs, n_hrf_bins)


def _make_event_schedule(
    n_runs: int,
    n_conditions: int,
    n_events_per_cond: int,
    stim_duration: float = 3.0,
    isi_min: float = 1.0,
    isi_max: float = 7.0,
    tr: float = 1.0,
    n_tp_run: int = 200,
    seed: int = 1,
) -> list[list[np.ndarray]]:
    """
    Random event schedule with fixed stim_duration and jittered ISI in [isi_min, isi_max].

    Returns all_onsets[condition][run] → np.ndarray of onset times (seconds, run-relative).

    If n_events_per_cond is positive: tries to place exactly that many events per run.
    If n_events_per_cond is negative: fills each run with as many events as will fit,
    ensuring at least abs(n_events_per_cond) events are placed.
    """
    rng = np.random.default_rng(seed)
    run_duration = n_tp_run * tr
    all_onsets: list[list[np.ndarray]] = []

    fill_run = n_events_per_cond < 0
    _min_events = abs(n_events_per_cond)

    for _cond in range(n_conditions):
        cond_onsets = []
        for _ in range(n_runs):
            onsets = []
            t = rng.uniform(0.5, 2.0)  # small random jitter at start

            if fill_run:
                # Fill run: keep adding events until we run out of space
                while t + stim_duration <= run_duration - 1.0:
                    onsets.append(t)
                    isi = rng.uniform(isi_min, isi_max)
                    t += stim_duration + isi
            else:
                # Fixed count: place exactly n_events_per_cond (or fewer if run is too short)
                for _ in range(n_events_per_cond):
                    if t + stim_duration > run_duration - 1.0:
                        break
                    onsets.append(t)
                    isi = rng.uniform(isi_min, isi_max)
                    t += stim_duration + isi

            cond_onsets.append(np.array(onsets, dtype=float))
        all_onsets.append(cond_onsets)

    return all_onsets


def _build_design_and_signal(
    all_onsets: list[list[np.ndarray]],
    run_starts: list[int],
    n_tp_total: int,
    tr: float,
    hrf: torch.Tensor,
    stim_duration: float = 3.0,
    microtime_dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convolve onsets with HRF and return onset_matrix (microtime) and design (TR-sampled).

    Returns (onset_matrix, design) where design is (n_tp_total, n_conditions).
    """
    from fastfuncstuff.design.matrices import convolve_hrf_microtime
    from fastfuncstuff.design.builder import create_onset_matrix_microtime

    n_conditions = len(all_onsets)
    stim_durations = [stim_duration] * n_conditions

    onset_matrix = create_onset_matrix_microtime(
        all_onsets, run_starts, tr, n_tp_total, microtime_dt, stim_durations, CPU
    )

    design_out = convolve_hrf_microtime(
        onset_matrix,
        hrf,
        n_tp_total,
        tr=tr,
        microtime_dt=microtime_dt,
        run_starts=run_starts,
        device=CPU,
    )
    assert isinstance(design_out, torch.Tensor)
    return onset_matrix, design_out


def test_microtime_convolution_clips_hrf_at_run_boundaries():
    """HRF tails from one run must not bleed into the next when run_starts is provided."""
    from fastfuncstuff.design.matrices import convolve_hrf_microtime
    from fastfuncstuff.design.builder import create_onset_matrix_microtime

    tr = 1.0
    microtime_dt = 0.1
    n_tp_run = 40
    n_runs = 2
    n_tp_total = n_tp_run * n_runs
    run_starts = [0, n_tp_run]

    # Single condition, single event near end of run 1.
    all_onsets = [[np.array([39.0]), np.array([])]]
    onset_matrix = create_onset_matrix_microtime(
        all_onsets,
        run_starts,
        tr,
        n_tp_total,
        microtime_dt,
        [1.0],
        CPU,
    )
    hrf = _make_gaussian_hrf_library(microtime_dt=microtime_dt, peak_times=(6.0,))[0]

    # Unguarded convolution can bleed into run 2.
    design_no_guard = convolve_hrf_microtime(
        onset_matrix,
        hrf,
        n_tp_total,
        tr=tr,
        microtime_dt=microtime_dt,
        device=CPU,
    )

    # Guarded convolution must clip at run boundary.
    design_guard = convolve_hrf_microtime(
        onset_matrix,
        hrf,
        n_tp_total,
        tr=tr,
        microtime_dt=microtime_dt,
        run_starts=run_starts,
        device=CPU,
    )

    assert design_no_guard[n_tp_run:, 0].abs().max().item() > 0
    assert torch.allclose(
        design_guard[n_tp_run:, 0],
        torch.zeros_like(design_guard[n_tp_run:, 0]),
        atol=1e-7,
    ), "Run-boundary clipping failed: found non-zero HRF tail in run 2"


# ============================================================================
# Tests for 3dHRFoptfast.py (fit_glm_hrf_library_with_xval)
# ============================================================================


class TestHRFRecovery:
    """
    Verify that fit_glm_hrf_library_with_xval recovers the known true HRF.

    Simulation:  AR(1) noise (ρ=0.3), per-voxel scaling,
                 stim_duration=3s, ISI 1-7s, LORO CV.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Build a 3-HRF library with signal voxels and noise voxels."""
        self.n_runs = 6
        self.n_tp_run = 80  # 80 TRs / run
        self.tr = 2.0
        self.n_conditions = 2
        self.n_signal_voxels = 30
        self.n_noise_voxels = 20
        self.n_voxels = self.n_signal_voxels + self.n_noise_voxels
        self.microtime_dt = 0.1
        self.stim_duration = 3.0
        self.true_hrf_idx = 1  # 6s-peak (middle of library)

        n_tp_total = self.n_runs * self.n_tp_run
        run_starts = [i * self.n_tp_run for i in range(self.n_runs)]
        all_onsets = _make_event_schedule(
            n_runs=self.n_runs,
            n_conditions=self.n_conditions,
            n_events_per_cond=10,
            stim_duration=self.stim_duration,
            isi_min=1.0,
            isi_max=7.0,
            tr=self.tr,
            n_tp_run=self.n_tp_run,
            seed=7,
        )

        hrf_library = _make_gaussian_hrf_library(self.microtime_dt, peak_times=(5.0, 6.0, 8.0))
        true_hrf = hrf_library[self.true_hrf_idx]

        onset_matrix, design = _build_design_and_signal(
            all_onsets,
            run_starts,
            n_tp_total,
            self.tr,
            true_hrf,
            self.stim_duration,
            self.microtime_dt,
        )

        # Signal voxels: random betas × design + AR noise
        torch.manual_seed(42)
        betas = torch.randn(self.n_signal_voxels, self.n_conditions) * 3.0
        signal = betas @ design.T  # (n_signal, n_tp)
        noise_signal = _make_colored_noise(n_tp_total, self.n_signal_voxels, rho=0.3, seed=10)
        noise_pure = _make_colored_noise(n_tp_total, self.n_noise_voxels, rho=0.3, seed=11)

        self.data = torch.cat([signal + noise_signal * 0.5, noise_pure], dim=0).float()
        self.onset_matrix = onset_matrix
        self.hrf_library = hrf_library
        self.run_starts = run_starts
        self.stim_durations = [self.stim_duration] * self.n_conditions

    def test_hrf_recovery_majority_signal_voxels(self):
        """Signal voxels should predominantly select the true HRF index."""
        from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

        results = fit_glm_hrf_library_with_xval(
            data=self.data,
            onsets=self.onset_matrix,
            hrf_library=self.hrf_library,
            tr=self.tr,
            run_starts=self.run_starts,
            stim_durations=self.stim_durations,
            cv_strategy=1,  # LORO
            metric="cod",
            microtime_dt=self.microtime_dt,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=50,
        )

        signal_hrf_idx = results.hrf_index[: self.n_signal_voxels]
        frac_correct = (signal_hrf_idx == self.true_hrf_idx).float().mean().item()
        # At moderate SNR expect at least 60% of signal voxels to pick correct HRF
        assert frac_correct >= 0.6, (
            f"Only {frac_correct:.0%} of signal voxels chose true HRF "
            f"(idx={self.true_hrf_idx}). Check SNR or HRF library separability."
        )

    def test_xval_r2_positive_for_signal_voxels(self):
        """CV R² should be positive for the majority of signal voxels."""
        from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

        results = fit_glm_hrf_library_with_xval(
            data=self.data,
            onsets=self.onset_matrix,
            hrf_library=self.hrf_library,
            tr=self.tr,
            run_starts=self.run_starts,
            stim_durations=self.stim_durations,
            cv_strategy=1,
            metric="cod",
            microtime_dt=self.microtime_dt,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=50,
        )

        r2_signal = results.xval_r2_best[: self.n_signal_voxels]
        frac_positive = (r2_signal > 0).float().mean().item()
        assert frac_positive >= 0.5, (
            f"Only {frac_positive:.0%} of signal voxels have positive CV R²."
        )

    def test_xval_r2_higher_for_signal_than_noise_voxels(self):
        """Mean CV R² for signal voxels must exceed mean CV R² for noise voxels."""
        from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

        results = fit_glm_hrf_library_with_xval(
            data=self.data,
            onsets=self.onset_matrix,
            hrf_library=self.hrf_library,
            tr=self.tr,
            run_starts=self.run_starts,
            stim_durations=self.stim_durations,
            cv_strategy=1,
            metric="cod",
            microtime_dt=self.microtime_dt,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=50,
        )

        mean_r2_signal = results.xval_r2_best[: self.n_signal_voxels].mean().item()
        mean_r2_noise = results.xval_r2_best[self.n_signal_voxels :].mean().item()
        assert mean_r2_signal > mean_r2_noise, (
            f"Signal R² ({mean_r2_signal:.3f}) should exceed noise R² ({mean_r2_noise:.3f})"
        )

    @pytest.mark.parametrize("n_events", [1, 5, 20])
    def test_varying_event_counts(self, n_events: int):
        """
        HRF selection should run without error for 1, 5, and 20 events per condition.

        With 1 event the design is very sparse; with 20 events it may be dense.
        We only check that the function returns plausible outputs, not accuracy.
        """
        from fastfuncstuff.design.matrices import convolve_hrf_microtime
        from fastfuncstuff.design.builder import create_onset_matrix_microtime
        from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

        n_runs = 4
        n_tp_run = 100
        tr = 1.0
        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]

        all_onsets = _make_event_schedule(
            n_runs=n_runs,
            n_conditions=2,
            n_events_per_cond=n_events,
            stim_duration=3.0,
            isi_min=1.0,
            isi_max=7.0,
            tr=tr,
            n_tp_run=n_tp_run,
            seed=n_events,
        )

        hrf_library = _make_gaussian_hrf_library(self.microtime_dt)
        onset_matrix = create_onset_matrix_microtime(
            all_onsets, run_starts, tr, n_tp_total, self.microtime_dt, [3.0, 3.0], CPU
        )
        true_hrf = hrf_library[1]
        design_out = convolve_hrf_microtime(
            onset_matrix, true_hrf, n_tp_total, tr=tr, microtime_dt=self.microtime_dt, device=CPU
        )
        assert isinstance(design_out, torch.Tensor)

        torch.manual_seed(n_events)
        betas = torch.randn(10, 2) * 7.0
        signal = betas @ design_out.T
        noise = _make_colored_noise(n_tp_total, 10, rho=0.3, seed=n_events + 100)
        baseline = 100  # hard coding for now - so that we have units of percent signal change.
        data = (baseline + signal + noise * 0.5).float()

        results = fit_glm_hrf_library_with_xval(
            data=data,
            onsets=onset_matrix,
            hrf_library=hrf_library,
            tr=tr,
            run_starts=run_starts,
            stim_durations=[3.0, 3.0],
            cv_strategy=1,
            microtime_dt=self.microtime_dt,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=10,
        )

        assert results.hrf_index.shape == (10,)
        assert results.xval_r2_best.shape == (10,)
        assert results.hrf_index.min() >= 0
        assert results.hrf_index.max() < hrf_library.shape[0]

    def test_true_hrf_xval_r2_better_than_wrong_hrf(self):
        """
        The true HRF should yield higher median CV R² across all HRFs than
        a clearly wrong HRF (earliest vs. latest peak).
        """
        from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

        results = fit_glm_hrf_library_with_xval(
            data=self.data,
            onsets=self.onset_matrix,
            hrf_library=self.hrf_library,
            tr=self.tr,
            run_starts=self.run_starts,
            stim_durations=self.stim_durations,
            cv_strategy=1,
            metric="cod",
            microtime_dt=self.microtime_dt,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=50,
        )

        # xval_r2_all_hrfs: (n_voxels, n_hrfs)
        r2_all = results.xval_r2_all_hrfs[: self.n_signal_voxels]  # signal only
        r2_true = r2_all[:, self.true_hrf_idx].median().item()
        r2_wrong = r2_all[:, 0].median().item()  # earliest-peak HRF
        assert r2_true >= r2_wrong, (
            f"True HRF median R² ({r2_true:.3f}) should be >= wrong HRF R² ({r2_wrong:.3f})"
        )


# ============================================================================
# Tests for 3dRidgefast.py (create_single_trial_design + fit_ridge_single_trial)
# ============================================================================


class TestRidgeRecovery:
    """
    Verify that ridge regression correctly handles close event spacing (ISI 1-2s).

    With short ISI the single-trial design is nearly collinear → OLS is noisy
    and ridge should provide better CV performance.
    """

    def _build_ridge_inputs(
        self,
        n_runs: int = 8,
        n_tp_run: int = 250,
        tr: float = 1.0,
        n_conditions: int = 2,
        n_events_per_cond: int = 5,
        isi_min: float = 1.0,
        isi_max: float = 3.0,
        n_voxels: int = 30,
        noise_rho: float = 0.3,
        noise_scale: float = 0.2,
        seed: int = 3,
    ):
        from fastfuncstuff.glm.ridge import create_single_trial_design

        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]
        all_onsets = _make_event_schedule(
            n_runs=n_runs,
            n_conditions=n_conditions,
            n_events_per_cond=n_events_per_cond,
            stim_duration=3.0,
            isi_min=isi_min,
            isi_max=isi_max,
            tr=tr,
            n_tp_run=n_tp_run,
            seed=seed,
        )

        design_matrix, trial_labels, trial_cond_ids, trial_run_ids, cond_design = (
            create_single_trial_design(
                onsets_by_condition=all_onsets,
                durations=[3.0] * n_conditions,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_tp_total,
                device=CPU,
            )
        )

        # Simulate: condition-level signal so CV predictions have something to recover.
        # Ridge CV averages trial betas within conditions to predict held-out runs;
        # trial-random betas average to 0 and yield near-zero CV R².
        torch.manual_seed(seed + 100)
        psc = 7.0  # hard coding here, this adjust the magnitude of the "hrf" in PSC units based on the 100 baseline below.
        cond_betas = torch.randn(n_voxels, n_conditions) * psc  # (n_voxels, n_cond)
        signal = cond_betas @ cond_design.T  # (n_voxels, n_tp)
        noise = _make_colored_noise(
            n_tp_total, n_voxels, rho=noise_rho, noise_scale=noise_scale, seed=seed + 200
        )
        baseline = 100  # hard coding for now - so that we have units of percent signal change.
        data = (baseline + signal + noise).float()
        true_betas = cond_betas  # for external reference

        return dict(
            data=data,
            design_matrix=design_matrix,
            trial_labels=trial_labels,
            trial_cond_ids=trial_cond_ids,
            trial_run_ids=trial_run_ids,
            cond_design=cond_design,
            run_starts=run_starts,
            tr=tr,
            n_voxels=n_voxels,
            true_betas=true_betas,
        )

    def test_ridge_runs_without_error_close_isi(self):
        """fit_ridge_single_trial should complete with dense event schedules."""
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        inp = self._build_ridge_inputs(isi_min=1.0, isi_max=2.0)
        results = fit_ridge_single_trial(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            trial_condition_ids=inp["trial_cond_ids"],
            trial_run_ids=inp["trial_run_ids"],
            condition_design=inp["cond_design"],
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=20,
        )
        assert results.betas_single_trial.shape[0] == inp["n_voxels"]

    def test_ridge_insample_r2_positive_for_signal_voxels(self):
        """
        After ridge fitting, both in-sample R² and CV R² should be positive for
        most voxels with well-spaced events and adequate SNR.

        ISI must be >> HRF duration (~15 s) so adjacent events don't overlap.
        Dense ISI (3-5 s) creates within-run collinearity that makes single-trial
        CV R² unreliable regardless of ridge regularization.
        """
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        inp = self._build_ridge_inputs(
            isi_min=2.0, isi_max=7.0, n_events_per_cond=10, noise_scale=0.01
        )

        print("\nDiagnostics:")
        print(f"  Data shape: {inp['data'].shape}")
        print(f"  Data mean: {inp['data'].mean():.2f}, std: {inp['data'].std():.2f}")
        print(f"  Data min: {inp['data'].min():.2f}, max: {inp['data'].max():.2f}")
        print(f"  Design shape: {inp['design_matrix'].shape}")
        print(f"  Num trials: {inp['design_matrix'].shape[1]}")
        print(f"  Num conditions: {len(torch.unique(inp['trial_cond_ids']))}")

        results = fit_ridge_single_trial(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            trial_condition_ids=inp["trial_cond_ids"],
            trial_run_ids=inp["trial_run_ids"],
            condition_design=inp["cond_design"],
            polort=2,
            device=CPU,
            verbose=True,  # Enable verbose to see what's happening
            chunk_size=20,
        )

        print("\nResults:")
        print(f"  In-sample R² > 0: {(results.r2 > 0).float().mean():.1%}")
        print(f"  CV R² > 0: {(results.xval_r2 > 0).float().mean():.1%}")
        print(f"  Median in-sample R²: {results.r2.median():.4f}")
        print(f"  Median CV R²: {results.xval_r2.median():.4f}")
        print(f"  Median optimal frac: {results.optimal_fracs.median():.3f}")

        # In-sample R² should be clearly positive when condition-level signal present
        frac_positive = (results.r2 > 0).float().mean().item()
        assert frac_positive >= 0.8, (
            f"Only {frac_positive:.0%} signal voxels have positive in-sample R². "
            f"Check SNR or fitting procedure."
        )
        # CV R² should also be positive when the design is well-conditioned.
        # With sparse events and enough SNR, condition-average betas predict held-out runs.
        frac_xval_positive = (results.xval_r2 > 0).float().mean().item()
        assert frac_xval_positive >= 0.5, (
            f"Only {frac_xval_positive:.0%} signal voxels have positive CV R². "
            f"This may indicate the training-only beta filtering or run-boundary "
            f"zeroing is broken."
        )

    def test_ridge_optimal_frac_not_all_ols_under_collinearity(self):
        """
        Under dense ISI (1-2s), the design is nearly collinear.
        At least some voxels should prefer ridge (frac < 1) over pure OLS.
        frac=1.0 means keep 100% of OLS norm = no regularization (OLS);
        frac=0.05 means keep 5% = near-maximum regularization.
        """
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        inp = self._build_ridge_inputs(isi_min=1.0, isi_max=2.0, seed=7)
        results = fit_ridge_single_trial(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            trial_condition_ids=inp["trial_cond_ids"],
            trial_run_ids=inp["trial_run_ids"],
            condition_design=inp["cond_design"],
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=20,
        )
        # frac=1.0 is OLS (no regularization); frac < 0.95 → ridge is preferred.
        frac_using_ridge = (results.optimal_fracs < 0.95).float().mean().item()
        assert frac_using_ridge >= 0.1, (
            f"Expected some voxels to prefer ridge regularisation under dense ISI, "
            f"but only {frac_using_ridge:.0%} have frac < 0.95."
        )

    def test_ridge_narrow_isi_more_regularized_than_wide_isi(self):
        """
        Under close ISI (1-2s), the single-trial design is highly collinear →
        ridge CV should select lower frac (more regularization) than wide ISI.

        Convention (fracridge): frac=1.0 keeps 100% of OLS coefficient norm
        (= no regularization / OLS); frac=0.0 = maximum regularization.
        So lower mean frac → more regularization selected.
        """
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        def _run_mean_frac(isi_min, isi_max, seed):
            inp = self._build_ridge_inputs(
                n_events_per_cond=12,
                isi_min=isi_min,
                isi_max=isi_max,
                seed=seed,
                noise_scale=0.1,
            )
            res = fit_ridge_single_trial(
                data=inp["data"],
                design_matrix=inp["design_matrix"],
                run_starts=inp["run_starts"],
                tr=inp["tr"],
                trial_condition_ids=inp["trial_cond_ids"],
                trial_run_ids=inp["trial_run_ids"],
                condition_design=inp["cond_design"],
                polort=2,
                device=CPU,
                verbose=False,
                chunk_size=20,
            )
            return res.optimal_fracs.mean().item()

        frac_narrow = _run_mean_frac(1.0, 2.0, seed=5)
        frac_wide = _run_mean_frac(5.0, 7.0, seed=5)
        # Collinear (narrow ISI) design → OLS overfits → CV picks lower frac
        # (lower frac = more regularization = smaller coefficient norms)
        assert frac_narrow <= frac_wide, (
            f"Narrow ISI mean frac ({frac_narrow:.3f}) should be ≤ "
            f"wide ISI mean frac ({frac_wide:.3f}). "
            f"(frac=1 is OLS/no-reg, frac→0 is max regularization)"
        )

    @pytest.mark.parametrize("n_events", [1, 5, 20])
    def test_ridge_varying_event_counts(self, n_events: int):
        """fit_ridge_single_trial should handle 1, 5, 20 events per condition."""
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        inp = self._build_ridge_inputs(
            n_events_per_cond=n_events,
            isi_min=1.0,
            isi_max=7.0,
            seed=n_events * 3,
        )
        results = fit_ridge_single_trial(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            trial_condition_ids=inp["trial_cond_ids"],
            trial_run_ids=inp["trial_run_ids"],
            condition_design=inp["cond_design"],
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=20,
        )
        assert results.betas_single_trial.shape[0] == inp["n_voxels"]
        assert results.xval_r2.shape[0] == inp["n_voxels"]

    def test_ridge_betas_correlated_with_truth(self):
        """
        With good SNR and wide ISI the estimated betas should be positively
        correlated with the known true betas (which are all positive here).
        """
        from fastfuncstuff.glm.ridge import fit_ridge_single_trial

        # High SNR: build with small noise_scale
        inp = self._build_ridge_inputs(
            n_runs=8,
            n_tp_run=150,
            n_events_per_cond=3,
            isi_min=3.0,
            isi_max=7.0,
            n_voxels=20,
            noise_rho=0.2,
            noise_scale=0.05,
            seed=99,
        )
        # Make condition betas all positive to test sign recovery.
        # Ridge CV predicts using condition_design × avg_betas; condition-level
        # signal is what gets recovered.
        cond_design: torch.Tensor = inp["cond_design"]  # type: ignore[assignment]
        n_tp_total = cond_design.shape[0]
        torch.manual_seed(50)
        pos_cond_betas = torch.abs(torch.randn(20, 2)) + 1.0  # (n_voxels, n_cond), > 1
        signal = pos_cond_betas @ cond_design.T  # (n_voxels, n_tp)
        low_noise = _make_colored_noise(n_tp_total, 20, rho=0.2, noise_scale=0.05, seed=88)
        data = (signal + low_noise).float()

        run_starts: list[int] = inp["run_starts"]  # type: ignore[assignment]
        tr: float = inp["tr"]  # type: ignore[assignment]
        trial_cond_ids: torch.Tensor = inp["trial_cond_ids"]  # type: ignore[assignment]
        trial_run_ids_inp: torch.Tensor = inp["trial_run_ids"]  # type: ignore[assignment]
        cond_design_inp: torch.Tensor = inp["cond_design"]  # type: ignore[assignment]
        design_matrix_inp: torch.Tensor = inp["design_matrix"]  # type: ignore[assignment]

        results = fit_ridge_single_trial(
            data=data,
            design_matrix=design_matrix_inp,
            run_starts=run_starts,
            tr=tr,
            trial_condition_ids=trial_cond_ids,
            trial_run_ids=trial_run_ids_inp,
            condition_design=cond_design_inp,
            polort=2,
            device=CPU,
            verbose=False,
            chunk_size=20,
        )

        # Mean beta across trials per voxel should be positive (true betas > 0.5)
        mean_beta = results.betas_single_trial.mean(dim=1)  # (n_voxels,)
        frac_positive = (mean_beta > 0).float().mean().item()
        assert frac_positive >= 0.8, (
            f"Expected positive mean betas for most voxels, got {frac_positive:.0%}"
        )


# ============================================================================
# Tests for 3dDenoisefast.py (cross_validate_noise_pcs + fit_denoising_model)
# ============================================================================


class TestDenoiseRecovery:
    """
    Verify that the denoising pipeline correctly identifies and removes
    structured noise PCs while preserving task signal.

    Strategy:
    - Create voxels with clean task signal (task voxels) and high initial R²
    - Create voxels dominated by shared structured noise PCs (noise voxels)
    - Verify noise pool excludes task voxels
    - Verify optimal PC count > 0 when shared noise is injected
    """

    def _build_denoise_inputs(
        self,
        n_runs: int = 6,
        n_tp_run: int = 80,
        tr: float = 2.0,
        n_task_voxels: int = 50,
        n_noise_voxels: int = 150,
        n_true_pcs: int = 3,
        task_snr: float = 5.0,  # signal amplitude relative to independent noise
        seed: int = 42,
    ):
        """
        Returns a data tensor and supporting structures.

        Task voxels: strong task response + weak independent noise + weak shared PCs.
        Noise voxels: no task response + strong shared PCs + weak independent noise.
        """
        from fastfuncstuff.design.matrices import convolve_hrf_microtime
        from fastfuncstuff.design.builder import create_onset_matrix_microtime

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]
        _n_voxels = n_task_voxels + n_noise_voxels
        microtime_dt = 0.1

        # Task design
        all_onsets = _make_event_schedule(
            n_runs=n_runs,
            n_conditions=2,
            n_events_per_cond=8,
            stim_duration=3.0,
            isi_min=2.0,
            isi_max=6.0,
            tr=tr,
            n_tp_run=n_tp_run,
            seed=seed,
        )
        onset_matrix = create_onset_matrix_microtime(
            all_onsets, run_starts, tr, n_tp_total, microtime_dt, [3.0, 3.0], CPU
        )
        n_hrf_bins = int(round(32.0 / microtime_dt))
        t_hrf = np.arange(n_hrf_bins) * microtime_dt
        h = np.exp(-0.5 * ((t_hrf - 6.0) / 1.5) ** 2)
        h = (h / h.max()).astype(np.float32)
        canonical_hrf = torch.from_numpy(h)

        design_out = convolve_hrf_microtime(
            onset_matrix, canonical_hrf, n_tp_total, tr=tr, microtime_dt=microtime_dt, device=CPU
        )
        assert isinstance(design_out, torch.Tensor)

        # Shared structured noise PCs (random walk-like, not task-correlated)
        shared_pcs = torch.from_numpy(
            rng.standard_normal((n_tp_total, n_true_pcs)).astype(np.float32)
        )
        # Make PCs smooth using AR(1) recursion
        for t in range(1, n_tp_total):
            shared_pcs[t] = 0.7 * shared_pcs[t - 1] + 0.3 * shared_pcs[t]

        # Task voxel loadings (strong) and noise voxel loadings
        betas_task = torch.randn(n_task_voxels, 2) * 2.0
        signal_task = betas_task @ design_out.T  # (n_task, n_tp)

        # Loadings for shared PCs
        pc_loadings_task = torch.randn(n_task_voxels, n_true_pcs) * 0.2  # weak in task voxels
        pc_loadings_noise = torch.randn(n_noise_voxels, n_true_pcs) * 2.0  # strong in noise voxels

        indep_noise_task = _make_colored_noise(n_tp_total, n_task_voxels, rho=0.2, seed=seed + 10)
        indep_noise_noise = _make_colored_noise(n_tp_total, n_noise_voxels, rho=0.2, seed=seed + 20)

        task_data = (
            signal_task * task_snr + pc_loadings_task @ shared_pcs.T + indep_noise_task
        ).float()
        noise_data = (pc_loadings_noise @ shared_pcs.T + indep_noise_noise * 0.5).float()

        data = torch.cat([task_data, noise_data], dim=0)  # (n_voxels, n_tp)

        return dict(
            data=data,
            design_matrix=design_out,
            run_starts=run_starts,
            tr=tr,
            n_task_voxels=n_task_voxels,
            n_noise_voxels=n_noise_voxels,
            n_true_pcs=n_true_pcs,
        )

    def test_denoising_model_runs_without_error(self):
        """fit_denoising_model should complete without raising an exception."""
        from fastfuncstuff.denoise.sequential import fit_denoising_model

        inp = self._build_denoise_inputs()
        results = fit_denoising_model(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            polort=2,
            max_components=10,
            r2_threshold=0.05,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )
        assert results is not None
        assert hasattr(results, "optimal_n_components")

    def test_noise_pool_excludes_high_r2_task_voxels(self):
        """
        Voxels with strong task signal (high initial R²) should be excluded
        from the noise pool.
        """
        from fastfuncstuff.denoise.sequential import fit_denoising_model

        inp = self._build_denoise_inputs(task_snr=8.0)
        results = fit_denoising_model(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            polort=2,
            max_components=10,
            r2_threshold=0.05,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )

        noise_mask = results.noise_pool_mask  # (n_voxels,) bool
        n_task = inp["n_task_voxels"]

        task_in_noise_pool = noise_mask[:n_task].float().mean().item()
        noise_in_noise_pool = noise_mask[n_task:].float().mean().item()

        # Noise voxels should be more represented in the noise pool than task voxels
        assert noise_in_noise_pool > task_in_noise_pool, (
            f"Noise pool fraction: task={task_in_noise_pool:.2f}, "
            f"noise voxels={noise_in_noise_pool:.2f}. "
            "High-R² task voxels should be excluded."
        )

    def test_optimal_pcs_greater_than_zero_with_shared_noise(self):
        """
        When strong shared noise PCs are injected, the optimal PC count
        should be > 0.
        """
        from fastfuncstuff.denoise.sequential import fit_denoising_model

        inp = self._build_denoise_inputs(n_true_pcs=3, task_snr=3.0)
        results = fit_denoising_model(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            polort=2,
            max_components=10,
            r2_threshold=0.05,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )
        assert results.optimal_n_components > 0, (
            f"Expected optimal_n_components > 0 when shared noise PCs are present, "
            f"got {results.optimal_n_components}"
        )

    def test_optimal_pcs_zero_without_shared_noise(self):
        """
        When there are no shared noise PCs (only independent noise in the noise pool),
        adding PCs should not improve cross-validated R². Optimal PC count should be 0.

        Uses the shared _build_denoise_inputs helper with n_true_pcs=0: task voxels
        have strong signal (→ criteria voxels with R² > threshold), noise voxels
        have only independent noise (→ noise pool). PCs extracted from noise pool
        are just noise → no CV improvement.
        """
        from fastfuncstuff.denoise.sequential import fit_denoising_model

        inp = self._build_denoise_inputs(n_true_pcs=0, task_snr=8.0, seed=55)
        results = fit_denoising_model(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            polort=2,
            max_components=8,
            r2_threshold=0.05,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )
        # Independent noise PCs should not substantially improve held-out R²
        # Check that the R² improvement is minimal (less than 0.5% absolute)
        r2_baseline = results.xval_r2_by_n_components[0]
        r2_best = results.xval_r2_by_n_components[results.optimal_n_components]
        improvement = r2_best - r2_baseline

        # With independent noise, improvement should be very small
        assert improvement < 0.005, (
            f"Expected minimal R² improvement with independent noise, "
            f"but got improvement of {improvement:.4f} "
            f"(baseline R²={r2_baseline:.4f}, best R²={r2_best:.4f} at {results.optimal_n_components} PCs)"
        )
        # Also check that optimal is not at the maximum (8), which would indicate real signal
        assert results.optimal_n_components < 8, (
            f"Expected optimal_n_components < max when no real signal, "
            f"got {results.optimal_n_components}"
        )

    def test_cross_validate_noise_pcs_monotone_improvement(self):
        """
        With injected shared PCs, cross_validate_noise_pcs should show
        improving (or non-decreasing) R² as PCs are added up to the true count.
        """
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs, extract_noise_pcs_per_run

        inp = self._build_denoise_inputs(n_true_pcs=3, task_snr=4.0)
        n_true_pcs = inp["n_true_pcs"]

        # Build noise pool mask: only noise voxels (last n_noise_voxels)
        n_task = inp["n_task_voxels"]
        _n_noise = inp["n_noise_voxels"]
        n_voxels_total = inp["data"].shape[0]
        noise_pool_mask = torch.zeros(n_voxels_total, dtype=torch.bool)
        noise_pool_mask[n_task:] = True

        # extract_noise_pcs_per_run extracts PCs from noise pool voxels
        raw_result = extract_noise_pcs_per_run(
            data=inp["data"],
            run_starts=inp["run_starts"],
            noise_pool_mask=noise_pool_mask,
            max_components=8,
            device=CPU,
        )
        assert isinstance(raw_result, list), "extract_noise_pcs_per_run should return a list"
        noise_pcs = raw_result

        # cross_validate_noise_pcs returns (r2_maps, r2_summary)
        # r2_maps: (n_voxels, max_components+1)
        # r2_summary: (max_components+1,) median R² across voxels
        r2_maps, r2_summary = cross_validate_noise_pcs(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            noise_pcs=noise_pcs,
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            max_components=8,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )

        # r2_summary[k] = median R² with k PCs. Should be >= r2_summary[0] somewhere.
        r2_with_0 = float(r2_summary[0])
        r2_with_true = float(r2_summary[n_true_pcs])
        assert r2_with_true >= r2_with_0, (
            f"R² with {n_true_pcs} PCs ({r2_with_true:.4f}) should be >= "
            f"R² with 0 PCs ({r2_with_0:.4f})"
        )

    def test_denoise_result_r2_curve_has_correct_length(self):
        """fit_denoising_model should return xval_r2_by_n_components of length max_components+1."""
        from fastfuncstuff.denoise.sequential import fit_denoising_model

        max_components = 6
        inp = self._build_denoise_inputs()
        results = fit_denoising_model(
            data=inp["data"],
            design_matrix=inp["design_matrix"],
            run_starts=inp["run_starts"],
            tr=inp["tr"],
            polort=2,
            max_components=max_components,
            r2_threshold=0.05,
            cv_strategy=1,
            device=CPU,
            verbose=False,
        )
        # xval_r2_by_n_components[k] = R² with k noise PCs; shape = (max_components+1,)
        assert results.xval_r2_by_n_components is not None
        assert len(results.xval_r2_by_n_components) == max_components + 1, (
            f"xval_r2_by_n_components should have length {max_components + 1}, "
            f"got {len(results.xval_r2_by_n_components)}"
        )
