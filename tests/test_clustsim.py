"""Tests for the 3dClustSim-style Monte-Carlo cluster null (stats/clustsim.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.clustsim import (
    ACF,
    DEFAULT_CS_ATHR,
    DEFAULT_CS_PTHR,
    LOTS_ATHR,
    NullFieldSimulator,
    acf_fwhm,
    acf_rfunc,
    acf_rfunc_inv,
    gumbel_extent_table,
    next_fft_size,
    random_field_grid,
    simulate_cluster_null,
    zthresholds,
)


def _ball_mask(nx=32, ny=32, nz=24):
    g = [(np.arange(n) - (n - 1) / 2) / ((n - 1) / 2) for n in (nx, ny, nz)]
    gx, gy, gz = np.meshgrid(*g, indexing="ij")
    return (gx**2 + gy**2 + gz**2) < 0.85**2


# ---------------------------------------------------------------------------
# ACF model
# ---------------------------------------------------------------------------


def test_acf_rejects_out_of_range_parameters():
    with pytest.raises(ValueError):
        ACF(a=1.5, b=2.0, c=3.0)
    with pytest.raises(ValueError):
        ACF(a=0.5, b=0.0, c=3.0)
    with pytest.raises(ValueError):
        ACF(a=0.5, b=2.0, c=-1.0)


def test_pure_gaussian_acf_fwhm_is_analytic():
    """a=1 reduces to exp(-r²/2b²), whose half-max radius is b·√(2ln2)."""
    acf = ACF(a=1.0, b=3.0, c=1.0)
    assert acf_fwhm(acf) == pytest.approx(2.0 * 3.0 * math.sqrt(2.0 * math.log(2.0)), rel=1e-4)


def test_rfunc_inv_inverts_rfunc():
    acf = ACF(a=0.6, b=3.0, c=5.0)
    for val in (0.8, 0.5, 0.2, 0.02):
        r = acf_rfunc_inv(val, acf)
        assert acf_rfunc(r, acf) == pytest.approx(val, abs=1e-6)


def test_from_fwhm_recovers_the_requested_smoothness():
    """Blurring white noise to FWHM f leaves an ACF of FWHM f·√2."""
    assert acf_fwhm(ACF.from_fwhm(6.0)) == pytest.approx(6.0 * math.sqrt(2.0), rel=1e-4)


# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------


def test_next_fft_size_is_afni_one35():
    for n in range(1, 400):
        v = next_fft_size(n)
        assert v >= n
        m = v
        for f in (3, 5):
            if m % f == 0:
                m //= f  # at most one factor of 3 and one of 5
        assert m & (m - 1) == 0, f"next_fft_size({n})={v} is not 2^p·3^q·5^r with q,r<=1"


def test_grid_matches_afni_reported_padding():
    """3dClustSim on this case prints '64x64x40 pads to 80x80x60'."""
    acf = ACF(a=0.6, b=3.0, c=5.0)
    assert random_field_grid((64, 64, 40), (2.5, 2.5, 2.5), acf) == (80, 80, 60)
    # ... and the ACF's own FWHM, which AFNI prints as 7.03.
    assert acf_fwhm(acf) == pytest.approx(7.03, abs=0.01)


def test_grid_floor_of_16_voxels():
    acf = ACF(a=1.0, b=0.5, c=0.5)
    assert min(random_field_grid((4, 4, 4), (3.0, 3.0, 3.0), acf)) >= 16


# ---------------------------------------------------------------------------
# Field generation
# ---------------------------------------------------------------------------


def test_generated_fields_are_unit_variance_and_zero_mean():
    mask = _ball_mask()
    sim = NullFieldSimulator(
        mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), device=torch.device("cpu"), seed=0
    )
    f = sim.generate(64)
    assert f.shape == (64, int(mask.sum()))
    # Normalisation is over the whole cropped grid, so the in-mask sd only has
    # to be close to 1, not exactly 1.
    assert float(f.std()) == pytest.approx(1.0, abs=0.05)
    assert float(f.mean()) == pytest.approx(0.0, abs=0.05)


def test_generated_field_has_the_requested_acf():
    """Round-trip through our own 3dFWHMx port: ask for an ACF, get it back."""
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    mask = _ball_mask(48, 48, 32)
    vox = (2.5, 2.5, 2.5)
    truth = ACF(0.6, 3.0, 5.0)
    sim = NullFieldSimulator(mask, vox, truth, device=torch.device("cpu"), seed=3)
    fields = sim.generate(96).T.contiguous()
    est = estimate_fwhmx_run(
        fields,
        torch.from_numpy(mask),
        mask.shape,
        vox,
        device=torch.device("cpu"),
        progress=False,
    )
    assert est.fwhm == pytest.approx(acf_fwhm(truth), rel=0.06)
    assert est.a == pytest.approx(truth.a, abs=0.08)
    assert est.b == pytest.approx(truth.b, rel=0.15)


def test_odd_field_count_is_supported():
    """Fields come in pairs (real/imag); an odd request must not over-return."""
    mask = _ball_mask(24, 24, 16)
    sim = NullFieldSimulator(
        mask, (3.0, 3.0, 3.0), ACF(1.0, 2.0, 1.0), device=torch.device("cpu"), seed=1
    )
    assert sim.generate(7).shape[0] == 7


def test_paired_fields_are_independent():
    """Real and imaginary halves of one transform must not be correlated."""
    mask = _ball_mask(32, 32, 24)
    sim = NullFieldSimulator(
        mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), device=torch.device("cpu"), seed=5
    )
    f = sim.generate(64).numpy()
    r = [np.corrcoef(f[i], f[i + 1])[0, 1] for i in range(0, 64, 2)]
    assert abs(float(np.mean(r))) < 0.05


# ---------------------------------------------------------------------------
# Thresholds and the alpha table
# ---------------------------------------------------------------------------


def test_zthresholds_split_the_tail_for_two_sided():
    z = zthresholds((0.05, 0.01), ("1-sided", "2-sided", "bi-sided"))
    assert z["1-sided"][0] == pytest.approx(1.6448536, abs=1e-5)
    # 2-sided and bi-sided cut at p/2 — they differ in whether opposite-sign
    # voxels may join a cluster, not in where the threshold sits.
    assert z["2-sided"][0] == pytest.approx(1.959964, abs=1e-5)
    np.testing.assert_allclose(z["2-sided"], z["bi-sided"])


def test_gumbel_table_brackets_the_empirical_survival_function():
    """A null whose max is always exactly 10 must threshold at ~10."""
    n = 1000
    sizes = np.full((n, 1), 10, dtype=np.int64)
    tab = gumbel_extent_table(sizes, (0.05,), n)
    assert 10.0 <= tab[0, 0] <= 11.0


def test_gumbel_table_is_monotone_in_both_axes():
    rng = np.random.default_rng(0)
    # Two pthr columns: the looser one yields systematically larger clusters.
    sizes = np.stack([rng.poisson(60, size=4000), rng.poisson(12, size=4000)], axis=1).astype(
        np.int64
    )
    tab = gumbel_extent_table(sizes, DEFAULT_CS_ATHR, 4000)
    # Stricter alpha => larger required cluster.
    assert np.all(np.diff(tab, axis=1) > 0)
    # Tighter pthr => smaller required cluster.
    assert np.all(tab[0] > tab[1])


def test_gumbel_table_handles_a_null_that_never_survives():
    """No suprathreshold voxel anywhere: any cluster at all is significant."""
    tab = gumbel_extent_table(np.zeros((500, 1), dtype=np.int64), (0.05,), 500)
    assert tab[0, 0] == 1.0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_simulate_cluster_null_shapes_and_ordering():
    mask = _ball_mask(32, 32, 24)
    null = simulate_cluster_null(
        mask,
        (3.0, 3.0, 3.0),
        ACF(0.5, 3.0, 4.0),
        n_iter=64,
        nns=(1, 3),
        sideds=("1-sided", "2-sided"),
        device=torch.device("cpu"),
        n_jobs=1,
        seed=2,
        verbose=False,
    )
    for key in (("1-sided", 1), ("1-sided", 3), ("2-sided", 1), ("2-sided", 3)):
        ext = null.max_extent[key]
        assert ext.shape == (64, len(DEFAULT_CS_PTHR))
        # Columns are in the caller's pthr order (loosest first), and a looser
        # threshold can never yield a smaller max cluster.
        assert np.all(np.diff(ext, axis=1) <= 0)
    # NN3 is a superset of NN1's connectivity, so its clusters are never smaller.
    assert np.all(null.max_extent[("1-sided", 3)] >= null.max_extent[("1-sided", 1)])


def test_parallel_and_serial_drivers_agree():
    """The worker pool must not reorder or drop iterations."""
    mask = _ball_mask(24, 24, 16)
    kw = dict(
        n_iter=48,
        nns=(1,),
        sideds=("1-sided",),
        pthr=(0.01, 0.001),
        device=torch.device("cpu"),
        seed=11,
        verbose=False,
    )
    a = simulate_cluster_null(mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), n_jobs=1, **kw)
    b = simulate_cluster_null(mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), n_jobs=3, **kw)
    np.testing.assert_array_equal(a.max_extent[("1-sided", 1)], b.max_extent[("1-sided", 1)])


def test_gumbel_table_is_forced_monotone():
    """Monte-Carlo noise can invert two adjacent cells; AFNI edits those out."""
    rng = np.random.default_rng(3)
    # Deliberately noisy: only 200 iterations over 3 nearly-identical columns.
    sizes = rng.poisson(20, size=(200, 3)).astype(np.int64)
    tab = gumbel_extent_table(sizes, LOTS_ATHR, 200)
    assert np.all(np.diff(tab, axis=1) >= 0), "row must not decrease as athr tightens"
    assert np.all(np.diff(tab, axis=0) <= 0), "column must not increase as pthr tightens"


def test_nodec_rounds_up_before_the_table_is_read():
    """-nodec must reach the NIML table too, not just the .1D formatting."""
    rng = np.random.default_rng(4)
    sizes = rng.poisson(40, size=(2000, 2)).astype(np.int64)
    plain = gumbel_extent_table(sizes, DEFAULT_CS_ATHR, 2000)
    rounded = gumbel_extent_table(sizes, DEFAULT_CS_ATHR, 2000, nodec=True)
    assert np.all(rounded == np.floor(plain + 0.951))
    assert np.all(rounded == np.round(rounded))
