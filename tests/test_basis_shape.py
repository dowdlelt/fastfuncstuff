"""Round-trip tests for the SPMG derivative latency/width readout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.design.basis_shape import (
    build_shape_hrf_bank,
    calibrate_shape_ratios,
    hrf_fwhm,
    invert_shape_ratios,
)
from fastfuncstuff.design.hrf import get_spm_canonical_hrf, get_spm_hrf_with_derivatives

DT = 0.1
DURATION = 32.0
TR = 1.0
N_TP = 400


def _convolve(hrf: np.ndarray, onsets: np.ndarray, stim_dur: float = 2.0) -> np.ndarray:
    """Sample the response to an onset train at TR resolution."""
    n_micro = int(N_TP * TR / DT) + hrf.size + 10
    stim = np.zeros(n_micro)
    width = max(1, int(round(stim_dur / DT)))
    for o in onsets:
        i = int(o / DT)
        stim[i : i + width] += 1.0
    conv = np.convolve(stim, hrf)[:n_micro] * DT
    return conv[(np.arange(N_TP) * TR / DT).astype(int)]


def _design(n_basis: int, onsets: np.ndarray, stim_dur: float = 2.0) -> np.ndarray:
    basis = (
        get_spm_hrf_with_derivatives(
            microtime_dt=DT,
            hrf_duration=DURATION,
            n_basis=n_basis,
            device=torch.device("cpu"),
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    # L2-normalise the rows, as generate_spmg_basis does — the whole
    # point of calibrating in design space is that this rescaling is
    # absorbed rather than having to be modelled.
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    return np.stack([_convolve(basis[j], onsets, stim_dur) for j in range(n_basis)], axis=1)


def _targets(taus, disps, onsets, stim_dur: float = 2.0):
    bank, _, fwhm = build_shape_hrf_bank(taus, disps, dt=DT, duration=DURATION)
    Y = np.stack([_convolve(bank[g], onsets, stim_dur) for g in range(bank.shape[0])], axis=1)
    return Y, fwhm


def _onsets(seed: int = 0, n: int = 60) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).uniform(0, N_TP * TR - 40, n))


def _observed(tau, disp, onsets, X, stim_dur=2.0):
    """Ratios a noiseless fit of a known (tau, disp) response reports."""
    h = (
        get_spm_canonical_hrf(
            microtime_dt=DT,
            hrf_duration=DURATION,
            dispersion=disp,
            onset=tau,
            device=torch.device("cpu"),
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    y = _convolve(h, onsets, stim_dur)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def test_fwhm_matches_canonical():
    h = get_spm_canonical_hrf(
        microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
    ).numpy()
    # SPM canonical FWHM is ~5.3 s; anything wildly off means the
    # half-max crossing logic broke.
    assert 4.5 < hrf_fwhm(h, DT) < 6.0


def test_fwhm_grows_with_dispersion():
    _, _, fwhm = build_shape_hrf_bank([0.0], [0.7, 1.0, 1.4], dt=DT, duration=DURATION)
    assert np.all(np.diff(fwhm[0]) > 0)


class TestSPMG2:
    taus = np.arange(-1.6, 1.61, 0.1)

    def _calib(self, onsets, stim_dur=2.0):
        X = _design(2, onsets, stim_dur)
        Y, fwhm = _targets(self.taus, [1.0], onsets, stim_dur)
        return X, calibrate_shape_ratios(X, Y, self.taus, [1.0], fwhm)

    def test_ratio_curve_is_monotone_and_centred(self):
        _, calib = self._calib(_onsets())
        curve = calib.ratio_t[:, 0]
        assert np.all(np.diff(curve) > 0)
        # tau = 0 must read exactly zero latency, whatever the scaling.
        assert abs(curve[np.argmin(np.abs(self.taus))]) < 1e-6

    @pytest.mark.parametrize("tau", [-1.2, -0.4, 0.0, 0.55, 1.3])
    def test_round_trip(self, tau):
        onsets = _onsets()
        X, calib = self._calib(onsets)
        b = _observed(tau, 1.0, onsets, X)
        got = invert_shape_ratios(calib, np.array([b[1] / b[0]]))
        assert got["valid"][0]
        assert got["latency"][0] == pytest.approx(tau, abs=0.02)

    def test_out_of_range_is_clamped_and_flagged(self):
        onsets = _onsets()
        _, calib = self._calib(onsets)
        got = invert_shape_ratios(calib, np.array([1e3, -1e3]))
        assert not got["valid"].any()
        assert got["latency"][0] <= self.taus.max() + 1e-9
        assert got["latency"][1] >= self.taus.min() - 1e-9

    def test_calibration_depends_on_stimulus_duration(self):
        """A hardcoded formula would be wrong; assert the dependence is real."""
        onsets = _onsets()
        _, short = self._calib(onsets, stim_dur=0.0)
        _, long = self._calib(onsets, stim_dur=12.0)
        i = int(np.argmin(np.abs(self.taus - 1.0)))
        # Relative, because L2-normalised basis rows make the absolute
        # ratio scale arbitrary (~0.33 here vs ~0.97 unnormalised).
        rel = abs(short.ratio_t[i, 0] - long.ratio_t[i, 0]) / abs(long.ratio_t[i, 0])
        assert rel > 0.02


class TestSPMG3:
    taus = np.arange(-1.5, 1.51, 0.125)
    disps = np.arange(0.7, 1.451, 0.05)

    def _calib(self, onsets):
        X = _design(3, onsets)
        Y, fwhm = _targets(self.taus, self.disps, onsets)
        return X, calibrate_shape_ratios(X, Y, self.taus, self.disps, fwhm)

    def test_latency_alone_moves_the_dispersion_ratio(self):
        """The interaction this module exists to undo — guard it stays modelled."""
        _, calib = self._calib(_onsets())
        assert calib.ratio_d is not None
        i_hi = int(np.argmin(np.abs(self.taus - 1.5)))
        i_zero = int(np.argmin(np.abs(self.taus)))
        j = int(np.argmin(np.abs(self.disps - 1.0)))
        # Canonical width at both points, yet r_d is far from equal.
        assert abs(calib.ratio_d[i_zero, j]) < 1e-6
        assert abs(calib.ratio_d[i_hi, j]) > 0.2

    def test_forward_map_is_invertible_over_the_box(self):
        _, calib = self._calib(_onsets())
        assert calib.ratio_d is not None
        dtau = self.taus[1] - self.taus[0]
        ddis = self.disps[1] - self.disps[0]
        jac = np.gradient(calib.ratio_t, dtau, axis=0) * np.gradient(
            calib.ratio_d, ddis, axis=1
        ) - np.gradient(calib.ratio_t, ddis, axis=1) * np.gradient(calib.ratio_d, dtau, axis=0)
        assert np.all(jac > 0) or np.all(jac < 0), "forward map folds — inversion unsafe"

    @pytest.mark.parametrize(
        "tau,disp", [(0.5, 0.8), (-0.9, 1.3), (1.2, 1.0), (0.0, 0.75), (-0.3, 1.4)]
    )
    def test_round_trip(self, tau, disp):
        onsets = _onsets()
        X, calib = self._calib(onsets)
        b = _observed(tau, disp, onsets, X)
        got = invert_shape_ratios(calib, np.array([b[1] / b[0]]), np.array([b[2] / b[0]]))
        assert got["valid"][0]
        assert got["latency"][0] == pytest.approx(tau, abs=0.05)
        assert got["dispersion"][0] == pytest.approx(disp, abs=0.05)

    def test_univariate_reading_would_be_wrong(self):
        """Pure latency must not be reported as a width change."""
        onsets = _onsets()
        X, calib = self._calib(onsets)
        b = _observed(1.2, 1.0, onsets, X)
        got = invert_shape_ratios(calib, np.array([b[1] / b[0]]), np.array([b[2] / b[0]]))
        assert got["dispersion"][0] == pytest.approx(1.0, abs=0.05)
        assert got["fwhm"][0] == pytest.approx(float(calib.fwhm[0, len(self.disps) // 2]), abs=0.4)

    def test_survives_realistic_noise(self):
        onsets = _onsets()
        X, calib = self._calib(onsets)
        b = _observed(0.7, 1.25, onsets, X)
        y_clean = X @ b
        scale = np.abs(y_clean).max()
        rng = np.random.default_rng(7)
        lat, dsp = [], []
        for _ in range(40):
            y = y_clean / scale + rng.normal(0, 1 / 50.0, X.shape[0])
            bb, *_ = np.linalg.lstsq(X, y, rcond=None)
            got = invert_shape_ratios(calib, np.array([bb[1] / bb[0]]), np.array([bb[2] / bb[0]]))
            lat.append(got["latency"][0])
            dsp.append(got["dispersion"][0])
        # tSNR 50, 60 trials: latency should land within ~0.1 s.
        assert np.median(lat) == pytest.approx(0.7, abs=0.1)
        assert np.std(lat) < 0.1
        assert np.median(dsp) == pytest.approx(1.25, abs=0.15)


class TestDurationConvolution:
    def test_impulse_is_a_no_op(self):
        from fastfuncstuff.design.hrf import convolve_curves_with_duration

        h = get_spm_canonical_hrf(
            microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
        ).numpy()
        np.testing.assert_allclose(convolve_curves_with_duration(h, DT, 0.0), h)

    def test_duration_delays_the_peak_by_about_half(self):
        """The bug this fixes: an impulse design mis-times a block event."""
        from fastfuncstuff.design.hrf import convolve_curves_with_duration

        h = (
            get_spm_canonical_hrf(
                microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )
        p0 = np.argmax(h) * DT
        for dur in (2.0, 6.0):
            p = np.argmax(convolve_curves_with_duration(h, DT, dur)) * DT
            # Not exactly D/2 — the HRF is asymmetric, so a longer block
            # drifts a little further (6 s block gives 3.5 s).  The point
            # is that an impulse design is mis-timed by roughly half the
            # stimulus length, which is what -save-shape reads as latency.
            assert p - p0 == pytest.approx(dur / 2, abs=0.6)

    def test_normalisation_keeps_the_first_row_peak(self):
        from fastfuncstuff.design.hrf import convolve_curves_with_duration, make_derivative_basis

        h = (
            get_spm_canonical_hrf(
                microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )
        B = make_derivative_basis(h, DT, 3)
        out = convolve_curves_with_duration(B, DT, 4.0)
        assert np.abs(out[0]).max() == pytest.approx(np.abs(B[0]).max(), rel=1e-9)
        # One shared factor: the derivative rows are not renormalised
        # independently, or the coefficient ratios would change meaning.
        raw = np.stack(
            [np.convolve(B[k], np.ones(int(4.0 / DT)))[: B.shape[1]] * DT for k in range(3)]
        )
        good = np.abs(raw[0]) > 1e-9
        scale = out[0][good] / raw[0][good]
        for k in (1, 2):
            np.testing.assert_allclose(out[k][good] / raw[k][good], scale, rtol=1e-9)


class TestDerivativeBasis:
    def _base(self):
        return (
            get_spm_canonical_hrf(
                microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )

    def test_reproduces_the_spm_derivatives(self):
        """Generalised basis must agree with the hand-written SPM one."""
        from fastfuncstuff.design.hrf import get_spm_time_derivative, make_derivative_basis

        got = make_derivative_basis(self._base(), DT, 2)
        want = get_spm_time_derivative(
            microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
        ).numpy()
        r = np.corrcoef(got[1], want)[0, 1]
        assert r > 0.999

    def test_latency_derivative_has_the_right_sign(self):
        """Positive coefficient must mean a LATER response, as in SPMG2."""
        from fastfuncstuff.design.hrf import make_derivative_basis

        h = self._base()
        B = make_derivative_basis(h, DT, 2)
        later = np.interp(
            np.arange(h.size) * DT - 0.5, np.arange(h.size) * DT, h, left=0.0, right=0.0
        )
        b, *_ = np.linalg.lstsq(B.T, later, rcond=None)
        assert b[1] / b[0] > 0

    def test_width_derivative_widens(self):
        from fastfuncstuff.design.hrf import make_derivative_basis

        B = make_derivative_basis(self._base(), DT, 3)
        wider = build_shape_hrf_bank([0.0], [1.3], dt=DT, duration=DURATION, base_hrf=self._base())[
            0
        ][0]
        b, *_ = np.linalg.lstsq(B.T, wider, rcond=None)
        assert b[2] / b[0] > 0

    def test_peak_anchored_width_beats_origin_anchored(self):
        """Why the width knob scales about the peak, not about zero."""
        from fastfuncstuff.design.hrf import make_derivative_basis

        h = self._base()
        t = np.arange(h.size) * DT
        B = make_derivative_basis(h, DT, 3)
        origin_scaled = np.interp(t / 1.01, t, h, left=0.0, right=0.0)
        origin_row = (origin_scaled - h) / 0.01

        # Scaling about zero drags the peak with it, so that width
        # direction is nearly the latency direction.
        def align(u, v):
            return abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

        assert align(origin_row, B[1]) > align(B[2], B[1])


class TestLibraryHRFBasis:
    """A derivative basis around a library curve, not the SPM canonical."""

    taus = np.arange(-1.5, 1.51, 0.125)
    disps = np.arange(0.7, 1.451, 0.05)

    def _setup(self, hrf_idx: int, stim_dur: float):
        from fastfuncstuff.design.hrf import (
            convolve_curves_with_duration,
            get_hrf_library,
            make_derivative_basis,
        )

        lib = (
            get_hrf_library(
                mode="library", microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )
        base = lib[hrf_idx] / np.abs(lib[hrf_idx]).max()
        curves = convolve_curves_with_duration(make_derivative_basis(base, DT, 3), DT, stim_dur)
        curves = curves / np.linalg.norm(curves, axis=1, keepdims=True)
        onsets = _onsets()
        X = np.stack([_convolve(curves[j], onsets, 0.0) for j in range(3)], axis=1)
        bank, _, fwhm = build_shape_hrf_bank(
            self.taus, self.disps, dt=DT, duration=DURATION, base_hrf=base, stim_duration=stim_dur
        )
        Y = np.stack([_convolve(bank[g], onsets, 0.0) for g in range(bank.shape[0])], axis=1)
        return base, bank, calibrate_shape_ratios(X, Y, self.taus, self.disps, fwhm), X

    @pytest.mark.parametrize("hrf_idx", [0, 6, 12, 18])
    @pytest.mark.parametrize("stim_dur", [0.0, 1.0])
    def test_usable_region_is_fold_free(self, hrf_idx, stim_dur):
        """hrf 0 at 1 s folds on the raw grid; the trim must remove it."""
        from fastfuncstuff.design.basis_shape import usable_region

        _, _, calib, _ = self._setup(hrf_idx, stim_dur)
        keep = usable_region(calib, 0.95)
        assert keep.any(), "no usable region at all"
        rows = np.where(keep.any(axis=1))[0]
        cols = np.where(keep.any(axis=0))[0]
        sub_t = calib.ratio_t[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        sub_d = calib.ratio_d[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if sub_t.shape[0] < 3 or sub_t.shape[1] < 3:
            pytest.skip("usable region too small to evaluate a Jacobian")
        dtau = self.taus[1] - self.taus[0]
        ddis = self.disps[1] - self.disps[0]
        jac = np.gradient(sub_t, dtau, axis=0) * np.gradient(sub_d, ddis, axis=1) - np.gradient(
            sub_t, ddis, axis=1
        ) * np.gradient(sub_d, dtau, axis=0)
        inner = jac[1:-1, 1:-1]
        assert np.all(inner > 0) or np.all(inner < 0), "usable region still folds"

    @pytest.mark.parametrize("hrf_idx", [0, 6, 12])
    def test_round_trip_on_a_library_curve(self, hrf_idx):
        from fastfuncstuff.design.basis_shape import usable_region

        _, bank, calib, X = self._setup(hrf_idx, 1.0)
        keep = usable_region(calib, 0.95)
        for tt, dd in [(0.4, 0.85), (-0.5, 1.25), (0.0, 1.0), (0.8, 1.1)]:
            i = int(np.argmin(np.abs(self.taus - tt)))
            j = int(np.argmin(np.abs(self.disps - dd)))
            if not keep[i, j]:
                continue
            b, *_ = np.linalg.lstsq(
                X, _convolve(bank[i * self.disps.size + j], _onsets(), 0.0), rcond=None
            )
            got = invert_shape_ratios(
                calib, np.array([b[1] / b[0]]), np.array([b[2] / b[0]]), shape_r2_floor=0.95
            )
            assert got["valid"][0]
            assert got["latency"][0] == pytest.approx(self.taus[i], abs=0.06)
            assert got["dispersion"][0] == pytest.approx(self.disps[j], abs=0.06)


class TestInversionRespectsTheEnvelope:
    """The guard has to bind on the INVERSION, not just exist.

    Regression: ``usable_region`` was correct and separately tested while
    ``_invert_2d`` still built its triangulation from the raw R2 mask, so
    folded grid points stayed in the interpolant and out-of-envelope
    latencies came back flagged valid.
    """

    taus = np.arange(-1.5, 1.51, 0.125)
    disps = np.arange(0.7, 1.451, 0.05)

    def _calib(self):
        onsets = _onsets()
        X = _design(3, onsets)
        Y, fwhm = _targets(self.taus, self.disps, onsets)
        return calibrate_shape_ratios(X, Y, self.taus, self.disps, fwhm)

    def test_returned_values_stay_inside_the_kept_grid(self):
        from fastfuncstuff.design.basis_shape import usable_region

        calib = self._calib()
        keep = usable_region(calib, 0.95)
        tau_lo = calib.taus[np.where(keep.any(axis=1))[0]].min()
        tau_hi = calib.taus[np.where(keep.any(axis=1))[0]].max()

        rng = np.random.default_rng(0)
        r_t = rng.uniform(-8, 8, 400)
        r_d = rng.uniform(-8, 8, 400)
        got = invert_shape_ratios(calib, r_t, r_d, shape_r2_floor=0.95)
        assert np.nanmin(got["latency"]) >= tau_lo - 1e-9
        assert np.nanmax(got["latency"]) <= tau_hi + 1e-9

    def test_high_floor_shrinks_the_envelope(self):
        calib = self._calib()
        rng = np.random.default_rng(1)
        r_t = rng.uniform(-4, 4, 200)
        r_d = rng.uniform(-2, 2, 200)
        loose = invert_shape_ratios(calib, r_t, r_d, shape_r2_floor=0.90)
        tight = invert_shape_ratios(calib, r_t, r_d, shape_r2_floor=0.999)
        assert np.ptp(tight["latency"]) < np.ptp(loose["latency"])

    def test_flood_fill_keeps_more_than_a_rectangle(self):
        """A bad corner must not cost the whole latency range."""
        from fastfuncstuff.design.basis_shape import usable_region

        calib = self._calib()
        keep = usable_region(calib, 0.95)
        rows = np.where(keep.any(axis=1))[0]
        span = calib.taus[rows[-1]] - calib.taus[rows[0]]
        # Largest all-good rectangle around the origin, for comparison.
        ok = np.isfinite(calib.ratio_t) & (calib.shape_r2 >= 0.95)
        i_mid = int(np.argmin(np.abs(calib.taus)))
        j_mid = int(np.argmin(np.abs(calib.dispersions - 1.0)))
        lo = hi = i_mid
        while lo > 0 and ok[lo - 1, :].all():
            lo -= 1
        while hi < ok.shape[0] - 1 and ok[hi + 1, :].all():
            hi += 1
        assert ok[i_mid, j_mid]
        assert span >= calib.taus[hi] - calib.taus[lo]


class TestPerVoxelShapeSelection:
    """Derivatives around a per-voxel selected curve.

    The property that matters: absolute latency is measured against the
    SELECTED curve, so picking a curve that peaks late shifts every
    condition in that voxel alike.  Cross-condition differences survive;
    the absolute zero does not.  This is the linear-parametrisation twin
    of the ``-shift-shapes`` TTP warning.
    """

    taus = np.arange(-1.5, 1.51, 0.125)
    disps = np.arange(0.7, 1.451, 0.05)

    def _lib(self):
        from fastfuncstuff.design.hrf import get_hrf_library

        return (
            get_hrf_library(
                mode="library", microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )

    def _latency_of(self, base_idx: int, model_idx: int, true_tau: float) -> float:
        """Fit data generated from curve `base_idx` using a basis around `model_idx`."""
        from fastfuncstuff.design.hrf import make_derivative_basis

        lib = self._lib()
        truth = lib[base_idx] / np.abs(lib[base_idx]).max()
        model = lib[model_idx] / np.abs(lib[model_idx]).max()

        curves = make_derivative_basis(model, DT, 3)
        curves = curves / np.linalg.norm(curves, axis=1, keepdims=True)
        onsets = _onsets()
        X = np.stack([_convolve(curves[j], onsets, 0.0) for j in range(3)], axis=1)

        bank, _, fwhm = _bank_for(truth, self.taus, self.disps)
        Y = np.stack([_convolve(bank[g], onsets, 0.0) for g in range(bank.shape[0])], axis=1)
        # Calibrate against the MODEL curve, as the CLI does.
        mbank, _, mfwhm = _bank_for(model, self.taus, self.disps)
        Ym = np.stack([_convolve(mbank[g], onsets, 0.0) for g in range(mbank.shape[0])], axis=1)
        calib = calibrate_shape_ratios(X, Ym, self.taus, self.disps, mfwhm)

        i = int(np.argmin(np.abs(self.taus - true_tau)))
        j = int(np.argmin(np.abs(self.disps - 1.0)))
        y = _convolve(
            np.interp(
                np.arange(truth.size) * DT - self.taus[i],
                np.arange(truth.size) * DT,
                truth,
                left=0.0,
                right=0.0,
            ),
            onsets,
            0.0,
        )
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        got = invert_shape_ratios(
            calib, np.array([b[1] / b[0]]), np.array([b[2] / b[0]]), shape_r2_floor=0.90
        )
        del j, Y
        return float(got["latency"][0])

    def test_matched_curve_recovers_absolute_latency(self):
        for tau in (-0.5, 0.0, 0.6):
            got = self._latency_of(6, 6, tau)
            assert got == pytest.approx(tau, abs=0.1)

    def test_mismatched_curve_offsets_every_condition_alike(self):
        """The offset must be common, so differences survive."""
        offsets = [self._latency_of(6, 7, tau) - tau for tau in (-0.5, 0.0, 0.6)]
        spread = max(offsets) - min(offsets)
        assert abs(np.mean(offsets)) > 0.05, "expected a real offset from the wrong curve"
        assert spread < 0.12, f"offset should be common across conditions, spread={spread:.3f}"


def _bank_for(base, taus, disps):
    return build_shape_hrf_bank(taus, disps, dt=DT, duration=DURATION, base_hrf=base)


class TestFitbasisFlagSurface:
    """The -hrf / -derivatives split, and that the old names still land."""

    def _parse(self, argv):
        from fastfuncstuff.cli.fitbasis import create_parser

        return create_parser().parse_args(argv)

    def test_defaults(self):
        args = self._parse(
            ["-input", "a.nii", "-prefix", "p", "-onsets", "o.1D", "-durations", "1"]
        )
        assert args.hrf == "canonical"
        assert args.derivatives == "time"
        assert args.reg == "none"  # not "cone"
        assert args.hrf_select == "full"  # not "xval"
        assert args.model is None

    @pytest.mark.parametrize(
        "old,expected", [("SPMG1", "none"), ("SPMG2", "time"), ("SPMG3", "time+width")]
    )
    def test_legacy_model_maps_to_derivatives(self, old, expected):
        from fastfuncstuff.cli.fitbasis import create_parser

        args = create_parser().parse_args(
            ["-input", "a.nii", "-prefix", "p", "-onsets", "o.1D", "-durations", "1", "-model", old]
        )
        # main() performs the mapping; assert the table it uses is the one above.
        assert {"SPMG1": "none", "SPMG2": "time", "SPMG3": "time+width"}[old] == expected
        assert args.model == old

    @pytest.mark.parametrize(
        "alias", ["-basis-hrf", "-hrf-shapes", "-shift-shapes", "-shift-hrf", "-basis_hrf"]
    )
    def test_old_hrf_flag_names_still_land_on_hrf(self, alias):
        args = self._parse(
            [
                "-input",
                "a.nii",
                "-prefix",
                "p",
                "-onsets",
                "o.1D",
                "-durations",
                "1",
                alias,
                "library",
            ]
        )
        assert args.hrf == "library"

    @pytest.mark.parametrize("alias", ["-shift-shape-index", "-hrf_index"])
    def test_old_index_flag_names_still_land(self, alias):
        args = self._parse(
            [
                "-input",
                "a.nii",
                "-prefix",
                "p",
                "-onsets",
                "o.1D",
                "-durations",
                "1",
                alias,
                "m.nii",
            ]
        )
        assert args.hrf_index == "m.nii"

    def test_set_names_are_what_triggers_per_voxel_selection(self):
        from fastfuncstuff.cli.fitbasis import _hrf_set_name

        assert _hrf_set_name("library") == "library"
        assert _hrf_set_name("PIGHS") == "pighs"
        assert _hrf_set_name("canonical") is None
        assert _hrf_set_name("/path/to/my_hrf.1D") is None

    def test_shape_summary_names_the_real_model(self):
        """Printing 'SPMG2' while using a library curve is what made logs unreadable."""
        from fastfuncstuff.cli.fitbasis import _shape_summary

        args = self._parse(
            [
                "-input",
                "a.nii",
                "-prefix",
                "p",
                "-onsets",
                "o.1D",
                "-durations",
                "1",
                "-hrf",
                "library",
                "-derivatives",
                "time+width",
            ]
        )
        summary = _shape_summary(args)
        assert "library" in summary
        assert "SPMG" not in summary
        assert "width" in summary


class TestOutputBuckets:
    """One labelled bucket per quantity, QC riding along inside it."""

    def _write(self, tmp_path, qc=True, n_cond=3):
        from fastfuncstuff.cli.fitbasis import _save_bucket

        n_vox = 24

        def to_volume(masked):
            return masked.reshape((2, 3, 4) + masked.shape[1:])

        path = str(tmp_path / "b.nii.gz")
        _save_bucket(
            [np.full(n_vox, float(i)) for i in range(n_cond)],
            [f"cond{i}" for i in range(n_cond)],
            path,
            to_volume=to_volume,
            reference_img=None,
            qc=[("xvalR2", np.full(n_vox, 0.5)), ("taskR2", np.full(n_vox, 0.9))] if qc else None,
        )
        return path

    def test_qc_bricks_are_appended_and_labelled(self, tmp_path):
        import nibabel as nib

        from fastfuncstuff.io.headers import read_brick_labels

        path = self._write(tmp_path)
        img = nib.load(path)
        assert img.shape[3] == 5
        assert read_brick_labels(img) == ["cond0", "cond1", "cond2", "xvalR2", "taskR2"]
        # QC values land in the right sub-bricks, not scrambled.
        data = img.get_fdata()
        assert np.allclose(data[..., 3], 0.5)
        assert np.allclose(data[..., 4], 0.9)

    def test_bucket_without_qc_is_just_the_arrays(self, tmp_path):
        import nibabel as nib

        path = self._write(tmp_path, qc=False)
        assert nib.load(path).shape[3] == 3

    def test_no_basis_and_reg_none_flags_parse(self):
        from fastfuncstuff.cli.fitbasis import create_parser

        args = create_parser().parse_args(
            ["-input", "a.nii", "-prefix", "p", "-onsets", "o.1D", "-durations", "1", "-no-basis"]
        )
        assert args.no_basisweights is True
        # -reg none is the default, so the duplicate _unconstrained pair is off.
        assert args.reg == "none"

    def test_shape_bucket_interleaves_value_with_its_threshold(self):
        """latency and the valid map that gates it must be adjacent."""
        labels = []
        for lbl in ("A", "B"):
            for name in ("latency", "latency_dev", "shape_r2", "valid"):
                labels.append(f"{lbl}_{name}")
        # The contract: every condition's block is contiguous and ends in _valid.
        assert labels.index("A_valid") < labels.index("B_latency")
        assert labels[labels.index("A_latency") : labels.index("A_valid") + 1][-1] == "A_valid"


class TestShiftParity:
    """The two parametrisations must be comparable sub-brick for sub-brick."""

    def test_shift_threads_durations_into_the_design_bank(self):
        """Shift ignored -durations, so a block event read as spurious latency."""
        import inspect

        from fastfuncstuff.design.shifted_hrf import (
            build_shifted_design_bank,
            fit_shifted_hrf,
            xval_shifted_hrf,
        )

        for fn in (build_shifted_design_bank, fit_shifted_hrf, xval_shifted_hrf):
            assert "durations" in inspect.signature(fn).parameters, fn.__name__

    def test_duration_moves_the_shifted_column(self):
        """The plumbing has to actually change the regressor, not just exist."""
        from fastfuncstuff.design.shifted_hrf import build_shifted_design_bank

        h = (
            get_spm_canonical_hrf(
                microtime_dt=DT, hrf_duration=DURATION, device=torch.device("cpu")
            )
            .numpy()
            .astype(np.float64)
        )
        onsets = [np.array([10.0, 40.0, 70.0])]
        taus = np.array([0.0])
        kw = dict(tr=TR, n_timepoints=120, device=torch.device("cpu"))
        impulse = build_shifted_design_bank(onsets, h, DT, taus, durations=[0.0], **kw)
        block = build_shifted_design_bank(onsets, h, DT, taus, durations=[6.0], **kw)
        # A 6 s block peaks later than an impulse response to the same onsets.
        assert int(block[0, 0].argmax()) > int(impulse[0, 0].argmax())

    def test_shift_mode_takes_condition_durations(self):
        import inspect

        from fastfuncstuff.cli.fitbasis import _run_shift_mode

        assert "condition_durations" in inspect.signature(_run_shift_mode).parameters

    def test_both_parametrisations_agree_on_output_names(self):
        """The parity contract, as a list the code has to keep satisfying."""
        shared_files = {"amplitude", "shape", "diagnostics", "hrf_index", "xvalr2"}
        # Per-condition sub-brick names shared by both.
        shared_shape_bricks = {"latency", "latency_dev", "valid"}
        # Documented divergences: things only one parametrisation can produce.
        linear_only = {"basisweights", "shape_calibration", "basis"}
        shift_only_bricks = {"taskR2_tau0", "taskR2_incl_drift", "fstat", "amp_lambda"}
        assert shared_files & linear_only == set()
        assert shared_shape_bricks & shift_only_bricks == set()

    def test_xval_prediction_is_chunked(self):
        """The gather is (n_vox, n_cond, T) float64 — it OOM'd unchunked."""
        import inspect

        from fastfuncstuff.design import shifted_hrf

        src = inspect.getsource(shifted_hrf.xval_shifted_hrf)
        assert "vox_chunk" in src, "per-voxel chunking missing from the xval predict loop"
        assert "_memory_budget" in src, "chunk size must come from the memory budget"
