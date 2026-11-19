"""Constant-modulus cardiac starts (physio_cm). CPU only, synthetic data."""

import numpy as np
import torch

from fastfuncstuff.processing import physio, physio_cm

CPU = torch.device("cpu")


def test_unwrap_around_rate_keeps_a_heartbeat_crossing_nyquist_continuous():
    """Aliased steps of a rate drifting through (k + 1/2)/TR sit near +-pi; plain
    unwrapping flips them by 2 pi (a cycle slip, a 1/TR rate jump); lifting around
    the local mean step must keep the drift."""
    tr = 2.0
    t = np.arange(300) * tr
    rate = 1.25 + 0.05 * np.sin(2 * np.pi * t / 120)  # crosses 2.5/TR repeatedly
    true_steps = 2 * np.pi * rate[:-1] * tr
    wrapped = np.angle(np.exp(1j * true_steps))
    lifted = physio_cm.unwrap_around_rate(wrapped, tr)
    # equal to the truth up to one common multiple of 2 pi
    offset = np.round((true_steps - lifted) / (2 * np.pi))
    assert np.all(offset == offset[0])
    # taking the steps as they come (what unwrapping the phase does) slips cycles
    assert len(np.unique(np.round((true_steps - wrapped) / (2 * np.pi)))) > 1


def test_constant_modulus_traces_find_a_rotating_pair_in_noise():
    """A phasor with an aliased, wandering phase mixed into 30 noisy series, among
    Gaussian sources of larger variance: the flat-radius search returns it first."""
    rng = np.random.default_rng(0)
    nt = 400
    phi = np.cumsum(rng.uniform(-np.pi, np.pi, nt) * 0.3 + 2.0)
    sources = np.vstack([np.cos(phi), np.sin(phi), 1.5 * rng.standard_normal((8, nt))])
    Y = rng.standard_normal((3000, sources.shape[0])) @ sources + 0.5 * rng.standard_normal(
        (3000, nt)
    )
    Y = (Y - Y.mean(1, keepdims=True)) / Y.std(1, keepdims=True)
    G = torch.from_numpy(Y.T @ Y)
    ratio, c = physio_cm.constant_modulus_traces(G)[0]
    assert ratio < 1.3
    lock = max(abs(np.exp(1j * (np.angle(c) - s * phi)).mean()) for s in (1, -1))
    assert lock > 0.9


def _folding_run(nt=260, shape=(16, 16, 24), tr=2.0, seed=3, heart=True):
    """TR 2 s, sequential slices, a pulse whose rate wanders 70-80/min: its alias
    folds at Nyquist (75/min) all run long, where spectral starts lose it."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    st = np.arange(nz) * tr / nz
    t = np.arange(nt)[None, :] * tr + st[:, None]
    fine = np.arange(0, nt * tr + 1, 0.01)
    rate = 1.25 + 0.08 * np.sin(2 * np.pi * fine / 70)
    clock = np.cumsum(rate) * 0.01 * 2 * np.pi
    vessel = rng.random(shape) < 0.05
    delay = rng.uniform(0, 0.15, shape)
    sign = np.where(rng.random(shape) < 0.7, 1.0, -1.0)
    theta = np.interp(t[None, None] - delay[..., None], fine, clock)
    beat = np.cos(theta) + 0.5 * np.cos(2 * theta - 0.5)
    pulse = 0.03 * sign[..., None] * beat * vessel[..., None] if heart else 0.0
    base = 100.0 + 20.0 * rng.random(shape)
    mag = base[..., None] * (1.0 + pulse + 0.01 * rng.standard_normal((nx, ny, nz, nt)))
    return mag.astype(np.float32), st, tr, (fine, clock)


def test_a_heartbeat_folding_at_nyquist_is_found_and_its_phase_reaches_the_regressors():
    mag, st, tr, (fine, clock) = _folding_run()
    traces = physio.excitation_traces(mag, st, tr, device=CPU)
    ref = physio.refine_cardiac(mag, traces, st, tr, "magnitude", split_axis=0, device=CPU)
    assert ref.found and ref.method == "constant_modulus", ref.tried
    traces.cardiac, traces.cardiac_phase = ref.trace, ref.phase
    _, _, diag = physio.slice_regressors(traces, "weighted", "magnitude")
    s = diag["series"]
    truth = np.interp(s["t_card"], fine, clock)
    assert abs(np.exp(1j * (s["phi_card"] - truth)).mean()) > 0.9


def test_without_a_heartbeat_the_constant_modulus_path_finds_nothing():
    mag, st, tr, _ = _folding_run(heart=False)
    traces = physio.excitation_traces(mag, st, tr, device=CPU)
    ref = physio.refine_cardiac(mag, traces, st, tr, "magnitude", split_axis=0, device=CPU)
    assert not ref.found


def test_batched_slice_alignments_equal_the_one_phase_estimator():
    """The batched null scorer must be the same estimator as slice_alignment of
    _voxel_fits, including an empty excitation and voxels with a non-zero mean."""
    rng = np.random.default_rng(0)
    nt, G = 200, 5
    times = np.arange(nt)[:, None] * 1.0 + np.arange(G)[None] / G
    Ys = [
        torch.from_numpy(rng.standard_normal((0 if g == 2 else 400 + 50 * g, nt)) + 3.0)
        for g in range(G)
    ]
    phases = [2 * np.pi * f * times + rng.uniform(0, 6) for f in rng.uniform(0.5, 2, 12)]
    got = physio.slice_alignments(Ys, phases, CPU)
    want = [physio.slice_alignment(physio._voxel_fits(Ys, q, CPU)) for q in phases]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_polar_retraction_returns_the_nearest_orthonormal_frame():
    M = torch.randn((6, 9, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    W = physio_cm._polar(M)
    eye = torch.eye(2, dtype=torch.float64).expand(6, 2, 2)
    torch.testing.assert_close(W.transpose(1, 2) @ W, eye)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)  # nearest frame = U V^T
    torch.testing.assert_close(W, U @ Vh)
