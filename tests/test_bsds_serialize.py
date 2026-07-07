"""Save/load a BSDS fit and decode new data from the reloaded model."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.model import (
    decode,
    fit_bsds,
    load_bsds_model,
    save_bsds_model,
)


def _make_sessions(seed=0, n=4, d=8, t=60):
    rng = np.random.default_rng(seed)
    return [torch.tensor(rng.standard_normal((d, t)), dtype=torch.float64) for _ in range(n)]


def test_save_load_roundtrip_decodes_identically(tmp_path):
    train = _make_sessions(0)
    held = _make_sessions(99, n=2)
    model = fit_bsds(train, n_states=4, max_ldim=3, n_init=2, n_iter=25, seed=0)

    path = str(tmp_path / "model.npz")
    save_bsds_model(model, path)
    loaded = load_bsds_model(path)

    # Fitted parameters survive the round-trip.
    torch.testing.assert_close(loaded.state_covs, model.state_covs)
    torch.testing.assert_close(loaded.state.wa, model.state.wa)
    assert loaded.n_states == model.n_states
    assert loaded.ldim == model.ldim

    # And decoding held-out data gives the same state paths as the in-memory model.
    dec_ref = decode(model, held)
    dec_loaded = decode(loaded, held)
    assert abs(dec_ref.loglik - dec_loaded.loglik) < 1e-6
    for a, b in zip(dec_ref.viterbi_states, dec_loaded.viterbi_states, strict=True):
        assert torch.equal(a, b)


def test_loaded_model_is_full_standin_for_qc(tmp_path):
    # A reloaded model must feed compute_state_stats/QC without refitting: the
    # training Viterbi/responsibilities and convergence info round-trip too.
    from fastfuncstuff.dynamics.states import compute_state_stats

    train = _make_sessions(1)
    model = fit_bsds(train, n_states=4, max_ldim=3, n_init=2, n_iter=25, seed=0)
    path = str(tmp_path / "m.npz")
    save_bsds_model(model, path)
    loaded = load_bsds_model(path)

    assert len(loaded.viterbi_states) == len(model.viterbi_states)
    for a, b in zip(loaded.viterbi_states, model.viterbi_states, strict=True):
        assert torch.equal(a, b)
    assert loaded.converged == model.converged
    assert len(loaded.objective_history) == len(model.objective_history)

    s_ref = compute_state_stats(model, tr=1.0)
    s_load = compute_state_stats(loaded, tr=1.0)
    import numpy as np

    np.testing.assert_allclose(s_load.group_occupancy, s_ref.group_occupancy)
    assert s_load.effective_state_count == s_ref.effective_state_count
