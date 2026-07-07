"""How reproducible is a BSDS fit across random initialisations?

BSDS inference is non-convex: independent k-means seeds land in different local
optima, so a single fit's occupancy or per-state FC could be a stable feature of
the data or an artifact of one lucky/unlucky start. This refits the same data
``n_repeats`` times from different seeds, aligns every repeat to a reference fit
by FC pattern (Hungarian match, :mod:`.bsds.fc_match`), and reports how
consistently each reference state's connectivity is recovered.

Read it as a companion to model selection: selection picks ``n_states``/``ldim``;
stability tells you which of the resulting states to *trust*. A state with high
mean matched-FC across repeats is a real, reproducible regime; a low or ragged
one is init-dependent and should be interpreted cautiously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.dynamics.bsds.fc_match import (
    fc_similarity_matrix,
    hungarian_match,
    model_state_covs,
    occupancy_from_viterbi,
)
from fastfuncstuff.dynamics.bsds.model import fit_bsds

# Stride between repeat seeds. fit_bsds consumes seeds fit_seed + [0, n_init) for
# its restarts (plus a few more for per-session k-means), so any stride comfortably
# larger than a realistic n_init + n_sessions keeps repeats' restart pools disjoint.
_SEED_STRIDE = 100003


@dataclass
class StateStabilityResult:
    """Cross-initialisation reproducibility of a BSDS fit's states."""

    n_repeats: int
    reference_occupancy: np.ndarray  # (K,) occupancy of the reference fit
    per_state_fc: np.ndarray  # (K,) mean matched-FC similarity across repeats
    per_state_fc_min: np.ndarray  # (K,) worst matched-FC across repeats
    matched_occupancy: np.ndarray  # (n_repeats-1, K) repeat occupancy, reference order
    mean_fc: float  # mean per-state FC over reference-occupied states
    occupancy_correlation: float  # mean occupancy corr (reference vs repeats)


def state_stability(
    sessions,
    n_states: int,
    max_ldim: int,
    *,
    n_repeats: int = 5,
    occ_threshold: float = 1e-3,
    seed: int = 0,
    show_progress: bool = False,
    **fit_kwargs,
) -> StateStabilityResult:
    """Refit ``sessions`` ``n_repeats`` times and score state reproducibility.

    Each repeat uses a well-separated ``seed`` (so the k-means init genuinely
    differs); repeat 0 is the reference. Every other repeat is Hungarian-matched
    to the reference by FC pattern, and the matched similarity per reference
    state is collected. ``**fit_kwargs`` (e.g. ``n_init``, ``n_iter``, ``device``)
    pass through to :func:`~fastfuncstuff.dynamics.bsds.model.fit_bsds`; keep the
    per-fit budget modest since this is ``n_repeats`` full fits.

    The repeat seeds are strided by a large constant rather than ``+1``: ``fit_bsds``
    uses restart seeds ``fit_seed + 0..n_init-1`` internally, so consecutive
    ``fit_seed`` values would overlap those windows and every repeat could pick the
    *same* best restart — reporting a spurious perfect agreement. The stride keeps
    each repeat's restart pool disjoint.
    """
    from tqdm.auto import tqdm

    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2 to compare fits")
    models = [
        fit_bsds(
            sessions,
            n_states=n_states,
            max_ldim=max_ldim,
            seed=seed + r * _SEED_STRIDE,
            **fit_kwargs,
        )
        for r in tqdm(range(n_repeats), desc="stability refits", disable=not show_progress)
    ]

    ref = models[0]
    ref_covs = model_state_covs(ref)
    ref_occ = occupancy_from_viterbi(ref.viterbi_states, n_states)

    fc_rows: list[np.ndarray] = []
    occ_rows: list[np.ndarray] = []
    occ_corrs: list[float] = []
    for m in models[1:]:
        sim = fc_similarity_matrix(ref_covs, model_state_covs(m))  # (K, K)
        row, col = hungarian_match(sim)  # square -> row is 0..K-1, col the permutation
        matched_fc = np.full(n_states, np.nan)
        matched_occ = np.full(n_states, np.nan)
        m_occ = occupancy_from_viterbi(m.viterbi_states, n_states)
        matched_fc[row] = sim[row, col]
        matched_occ[row] = m_occ[col]
        fc_rows.append(matched_fc)
        occ_rows.append(matched_occ)
        both = (ref_occ > occ_threshold) & (matched_occ > occ_threshold)
        if both.sum() > 1:
            occ_corrs.append(float(np.corrcoef(ref_occ[both], matched_occ[both])[0, 1]))

    fc_arr = np.stack(fc_rows)  # (n_repeats-1, K)
    per_state_fc = np.nanmean(fc_arr, axis=0)
    per_state_fc_min = np.nanmin(fc_arr, axis=0)
    occupied = ref_occ > occ_threshold
    mean_fc = float(np.nanmean(per_state_fc[occupied])) if occupied.any() else float("nan")
    occ_corr = float(np.mean(occ_corrs)) if occ_corrs else float("nan")

    return StateStabilityResult(
        n_repeats=n_repeats,
        reference_occupancy=ref_occ,
        per_state_fc=per_state_fc,
        per_state_fc_min=per_state_fc_min,
        matched_occupancy=np.stack(occ_rows),
        mean_fc=mean_fc,
        occupancy_correlation=occ_corr,
    )
