"""Unified interface over dynamic-state backends.

The native BSDS fit is one way to get discrete brain states + dynamic FC from ROI
time series; osl-dynamics (OHBA) offers others (a plain Gaussian HMM, DyNeMo's
RNN temporal model). This module normalises them behind a single call returning a
common schema, so downstream analysis (state stats, matching, CEBRA overlay)
doesn't care which produced the states.

The ``bsds`` backend is native and always available. The ``osl-hmm`` /
``osl-dynemo`` backends lazily import ``osl_dynamics`` (a heavy TensorFlow
dependency, part of the optional ``[dynamics]`` extra); they raise a clear
ImportError when it isn't installed and never affect the native path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DynamicStatesResult:
    """Backend-agnostic result: discrete states + their generative parameters."""

    n_states: int
    backend: str
    state_timecourse: list[np.ndarray]  # per session, MAP/most-likely state (T_i,)
    responsibilities: list[np.ndarray]  # per session, (T_i, K) state probabilities
    transition: np.ndarray  # (K, K)
    state_covariances: np.ndarray  # (K, D, D) — dynamic FC per state
    state_means: np.ndarray | None  # (K, D) or None
    extra: dict = field(default_factory=dict)  # backend-native objects


def _sessions_to_numpy_time_major(sessions) -> list[np.ndarray]:
    """Convert ``(D, N)`` torch/numpy sessions to ``(N, D)`` numpy (channels last)."""
    out = []
    for s in sessions:
        arr = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        out.append(np.ascontiguousarray(arr.T, dtype=np.float32))
    return out


def _fit_bsds_backend(sessions, n_states, **kwargs) -> DynamicStatesResult:
    from fastfuncstuff.dynamics.bsds.model import fit_bsds

    model = fit_bsds(sessions, n_states=n_states, **kwargs)
    return DynamicStatesResult(
        n_states=model.n_states,
        backend="bsds",
        state_timecourse=[v.cpu().numpy() for v in model.viterbi_states],
        responsibilities=[r.cpu().numpy() for r in model.responsibilities],
        transition=model.transition.cpu().numpy(),
        state_covariances=model.state_covs.cpu().numpy(),
        state_means=model.state_means.cpu().numpy(),
        extra={"model": model},
    )


def _fit_osl_backend(sessions, n_states, model_kind: str, **kwargs) -> DynamicStatesResult:
    """Best-effort adapter for osl-dynamics HMM / DyNeMo (lazy import)."""
    try:
        from osl_dynamics.data import Data  # ty: ignore[unresolved-import]
        from osl_dynamics.inference import modes  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "osl backends need osl-dynamics: `pip install osl-dynamics` "
            "(or the `[dynamics]` extra). The native `bsds` backend needs no extra deps."
        ) from exc

    time_major = _sessions_to_numpy_time_major(sessions)
    data = Data(time_major)
    n_channels = time_major[0].shape[1]

    if model_kind == "hmm":
        from osl_dynamics.models.hmm import Config, Model  # ty: ignore[unresolved-import]

        config = Config(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=kwargs.get("sequence_length", 200),
            learn_means=kwargs.get("learn_means", True),
            learn_covariances=True,
            batch_size=kwargs.get("batch_size", 16),
            learning_rate=kwargs.get("learning_rate", 1e-3),
            n_epochs=kwargs.get("n_epochs", 20),
        )
    else:  # dynemo
        from osl_dynamics.models.dynemo import Config, Model  # ty: ignore[unresolved-import]

        config = Config(
            n_modes=n_states,
            n_channels=n_channels,
            sequence_length=kwargs.get("sequence_length", 200),
            inference_n_units=kwargs.get("inference_n_units", 64),
            inference_normalization="layer",
            model_n_units=kwargs.get("model_n_units", 64),
            model_normalization="layer",
            learn_alpha_temperature=True,
            initial_alpha_temperature=1.0,
            learn_means=kwargs.get("learn_means", True),
            learn_covariances=True,
            batch_size=kwargs.get("batch_size", 16),
            learning_rate=kwargs.get("learning_rate", 1e-3),
            n_epochs=kwargs.get("n_epochs", 20),
        )

    model = Model(config)
    model.fit(data)
    alphas = model.get_alpha(data)  # list of (T_i, K)
    if not isinstance(alphas, list):
        alphas = [alphas]
    covariances = np.asarray(model.get_covariances())
    means = None
    try:
        means = np.asarray(model.get_means())
    except Exception:  # pragma: no cover - some configs don't learn means
        means = None

    timecourse = [modes.argmax_time_courses(a).argmax(axis=1) for a in alphas]
    trans = np.asarray(getattr(model, "trans_prob", np.eye(n_states)))
    return DynamicStatesResult(
        n_states=n_states,
        backend=f"osl-{model_kind}",
        state_timecourse=[np.asarray(t) for t in timecourse],
        responsibilities=[np.asarray(a) for a in alphas],
        transition=trans if trans.shape == (n_states, n_states) else np.eye(n_states),
        state_covariances=covariances,
        state_means=means,
        extra={"model": model, "data": data},
    )


def fit_dynamic_states(
    sessions,
    n_states: int,
    *,
    backend: str = "bsds",
    **kwargs,
) -> DynamicStatesResult:
    """Fit a dynamic-state model with the chosen backend, returning a common schema.

    ``backend`` is ``"bsds"`` (native), ``"osl-hmm"``, or ``"osl-dynemo"``.
    Backend-specific keyword arguments pass straight through.
    """
    if backend == "bsds":
        return _fit_bsds_backend(sessions, n_states, **kwargs)
    if backend == "osl-hmm":
        return _fit_osl_backend(sessions, n_states, "hmm", **kwargs)
    if backend == "osl-dynemo":
        return _fit_osl_backend(sessions, n_states, "dynemo", **kwargs)
    raise ValueError(f"unknown backend: {backend!r} (choose bsds, osl-hmm, osl-dynemo)")
