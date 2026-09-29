"""
Design Efficiency and Power Metrics

Implementation of Liu & Frank (2004) theory for quantifying:
1. Estimation efficiency (ability to estimate HRF shape)
2. Detection power (ability to detect activation amplitude)
3. Conditional entropy (design randomness)
4. Efficiency-power trade-offs

Core insight: Cannot maximize both efficiency and power simultaneously.
Must choose based on experimental goals.

Reference:
Liu, T. T., & Frank, L. R. (2004). Efficiency, power, and entropy in
event-related fMRI with multiple trial types. Part I: Theory.
NeuroImage, 21(1), 387-400.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_design_matrix_for_condition(
    onsets: torch.Tensor,
    condition_idx: int,
    n_timepoints: int,
    mode: str = "onoff",
    hrf_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extract design matrix for a single condition

    Parameters
    ----------
    onsets : torch.Tensor, shape (n_timepoints, n_conditions)
        Binary onset matrix
    condition_idx : int
        Which condition to extract
    n_timepoints : int
        Total timepoints
    mode : str, default='onoff'
        'onoff': Binary onsets (block/impulse), shape (n_timepoints, 1)
        'fir': Lagged design, shape (n_timepoints, hrf_length). Column ``lag``
            is the onset train shifted down by ``lag`` samples, so
            ``X[t, lag] = onsets[t - lag]``.
    hrf_length : int, optional
        Number of lags. Required for mode='fir'.
    device : torch.device, optional
        Device for computation

    Returns
    -------
    X_k : torch.Tensor, shape (n_timepoints, hrf_length) for FIR or (n_timepoints, 1) for onoff
        Design matrix for condition k

    Notes
    -----
    ``mode='fir'`` previously returned the onset column unchanged -- identical to
    ``'onoff'`` -- so the two modes were indistinguishable. Nothing called it
    (both metrics built their own lagged matrix inline), which is why the stub
    went unnoticed; those inline copies now call this.

    Onset values are used as given rather than thresholded, so a parametrically
    modulated regressor carries its amplitudes through.
    """
    if device is None:
        device = onsets.device

    onsets_k = onsets[:, condition_idx].to(device)

    if mode == "onoff":
        return onsets_k[:, None]
    elif mode == "fir":
        if hrf_length is None:
            raise ValueError("hrf_length is required for mode='fir'")
        X_k = torch.zeros((n_timepoints, hrf_length), device=device, dtype=onsets_k.dtype)
        usable = min(n_timepoints, onsets_k.shape[0])
        for lag in range(hrf_length):
            if lag >= n_timepoints:
                break
            X_k[lag:usable, lag] = onsets_k[: usable - lag]
        return X_k
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _baseline_nuisance(
    n_timepoints: int,
    poly_degree: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor | None:
    """Legendre-style polynomial nuisance columns, degree 0 = the baseline."""
    if poly_degree < 0:
        return None
    t = torch.linspace(-1.0, 1.0, n_timepoints, device=device, dtype=dtype)
    columns = [torch.ones_like(t)]
    for degree in range(1, poly_degree + 1):
        columns.append(t**degree)
    basis = torch.stack(columns, dim=1)
    q, _ = torch.linalg.qr(basis)
    return q


def _ar1_whiten(M: torch.Tensor, rho: float) -> torch.Tensor:
    """Apply the exact AR(1) whitener V^(-1/2) to the rows of ``M``.

    Prais-Winsten form: the first row is scaled by sqrt(1 - rho^2) and every
    later row has rho times its predecessor subtracted, so W V W^T = I for
    V[i, j] = rho^|i-j|. This is the Sigma^(-1/2) of Liu & Frank (2004) Eq. 2.
    """
    if rho == 0.0:
        return M
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    out = torch.empty_like(M)
    out[0] = M[0] * np.sqrt(1.0 - rho**2)
    out[1:] = M[1:] - rho * M[:-1]
    return out


def _condition_onsets(design: torch.Tensor, n_conditions: int, hrf_length: int) -> torch.Tensor:
    """The (N, Q) stimulus pattern, from an onset matrix or a stacked FIR matrix."""
    if design.shape[1] == n_conditions:
        return design
    if design.shape[1] == n_conditions * hrf_length:
        return design[:, ::hrf_length]
    # An onset matrix with extra trailing columns; conditions beyond it are empty.
    cols = [
        design[:, k] if k < design.shape[1] else torch.zeros_like(design[:, 0])
        for k in range(n_conditions)
    ]
    return torch.stack(cols, dim=1)


def _whitened_detrended(
    X: torch.Tensor, poly_degree: int, rho: float, nuisance: torch.Tensor | None = None
) -> torch.Tensor:
    """X_perp = P_S~ Sigma^(-1/2) X, Liu & Frank (2004) Eq. 2.

    The nuisance model is whitened along with the design before it is projected
    out; projecting first and whitening second is only approximately the same.
    """
    n = X.shape[0]
    S = _baseline_nuisance(n, poly_degree, X.device, X.dtype)
    if nuisance is not None:
        nuisance = nuisance.to(device=X.device, dtype=X.dtype)
        S = nuisance if S is None else torch.cat([S, nuisance], dim=1)
    Xw = _ar1_whiten(X, rho)
    if S is None or S.shape[1] == 0:
        return Xw
    q, _ = torch.linalg.qr(_ar1_whiten(S, rho))
    return Xw - q @ (q.T @ Xw)


def _contrast_variances(
    fisher: torch.Tensor, contrasts: list[torch.Tensor], rtol: float = 1e-10
) -> list[float]:
    """Tr[L F^+ L^T] per contrast, with ``inf`` for a contrast the design cannot estimate.

    A ridge on F (as this module once added) gives every contrast a finite
    variance, so two conditions with identical timing -- which no GLM can tell
    apart -- scored the same power as a well-separated pair. Estimability is
    decided instead: L is estimable iff it lies in the row space of F.
    """
    evals, evecs = torch.linalg.eigh(fisher)
    keep = evals > rtol * evals.abs().max().clamp_min(1e-300)
    V = evecs[:, keep]
    inv = (V / evals[keep]) @ V.T
    out = []
    for L in contrasts:
        residual = L - (L @ V) @ V.T
        if residual.norm() > 1e-6 * L.norm().clamp_min(1e-300):
            out.append(float("inf"))
        else:
            out.append(float(torch.trace(L @ inv @ L.T)))
    return out


def _condition_and_pairwise_contrasts(
    n_conditions: int,
) -> list[tuple[tuple[int, int], torch.Tensor]]:
    """D_ij of Liu & Frank Eq. 3: each trial type (i == j), then each pairwise difference."""
    out = []
    for i in range(n_conditions):
        for j in range(i, n_conditions):
            d = torch.zeros(1, n_conditions, dtype=torch.float64)
            d[0, i] = 1.0
            if j != i:
                d[0, j] = -1.0
            out.append(((i, j), d))
    return out


def compute_estimation_efficiency(
    design: torch.Tensor | np.ndarray,
    n_conditions: int,
    hrf_length: int,
    tr: float = 1.0,
    normalize: bool = True,
    device: torch.device | None = None,
    poly_degree: int = 0,
    rho: float = 0.0,
    nuisance: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Estimation efficiency for HRF shape, Liu & Frank (2004) Eqs. 4-5.

    The Q trial types enter one joint FIR design X = [X_1 ... X_Q] (N x kQ), is
    whitened and has the nuisance model projected out (Eq. 2), and the
    covariance of every contrast is read off the *joint* Fisher information:

        C_ij = L_ij (X_perp^T X_perp)^-1 L_ij^T,    L_ij = D_ij (x) I_k

    for each trial type (i == j) and each pairwise difference (i < j). Then

        xi_ij  = 1 / Tr[C_ij]
        xi_tot = 1 / mean_{i<=j} Tr[C_ij]                       (Eq. 5)

    Everything is in units of 1 / sigma^2: multiply Tr[C] by the noise variance
    to get the summed variance of the k HRF estimates. Efficiency grows with N,
    as it should -- a longer scan estimates better.

    Conditions are not scored in isolation. Doing so ignores the cross-terms of
    the Fisher information, which is where the collinearity between trial types
    lives; two conditions with identical timing are not separately estimable at
    all and score 0 here.

    Parameters
    ----------
    design : array-like, shape (n_timepoints, n_conditions)
        Stimulus pattern per trial type on the TR grid (a stacked FIR matrix of
        width n_conditions * hrf_length is also accepted; its first lag of each
        block is used).
    n_conditions : int
        Number of trial types Q.
    hrf_length : int
        Number of FIR lags k.
    tr : float
        Unused; kept for call compatibility.
    normalize : bool
        Also report efficiency as a fraction of the Eq. 26 bound
        N / (2 (Q + 1) k), which is exact for binary patterns in white noise
        with a constant nuisance term and approximate otherwise.
    poly_degree : int
        Legendre nuisance degree (0 = baseline only, -1 = none).
    rho : float
        AR(1) noise autocorrelation; the design is whitened with it.
    nuisance : array-like, optional
        Extra (N, l) nuisance columns (motion, per-run baselines, ...).

    Returns
    -------
    dict with
        'per_condition' : tensor (Q,), 1 / Tr[C_ii]
        'per_contrast'  : {(i, j): 1 / Tr[C_ij]} including i == j
        'total'         : xi_tot (Eq. 5)
        'mean'          : mean of 'per_condition'
        'normalized', 'mean_normalized', 'total_normalized' when normalize=True
    """
    if device is None:
        device = torch.device("cpu")
    design = torch.as_tensor(design, device=device).to(torch.float64)
    nuisance_t = None if nuisance is None else torch.as_tensor(nuisance, device=device)

    n_timepoints = design.shape[0]
    onsets = _condition_onsets(design, n_conditions, hrf_length)
    X = torch.cat(
        [
            compute_design_matrix_for_condition(
                onsets, k, n_timepoints, mode="fir", hrf_length=hrf_length, device=device
            )
            for k in range(n_conditions)
        ],
        dim=1,
    )
    X_perp = _whitened_detrended(X, poly_degree, rho, nuisance_t)
    fisher = X_perp.T @ X_perp

    eye = torch.eye(hrf_length, dtype=torch.float64, device=device)
    pairs = _condition_and_pairwise_contrasts(n_conditions)
    traces = _contrast_variances(fisher, [torch.kron(d.to(device), eye) for _, d in pairs])
    per_contrast = {
        ij: (0.0 if np.isinf(v) else 1.0 / v) for (ij, _), v in zip(pairs, traces, strict=True)
    }

    per_condition = torch.tensor([per_contrast[(k, k)] for k in range(n_conditions)])
    mean_trace = float(np.mean(traces))
    total = 0.0 if np.isinf(mean_trace) else 1.0 / mean_trace

    result: dict[str, Any] = {
        "per_condition": per_condition,
        "per_contrast": per_contrast,
        "total": total,
        "mean": per_condition.mean().item(),
    }
    if normalize:
        bound = n_timepoints / (2.0 * (n_conditions + 1) * hrf_length)
        result["normalized"] = per_condition / bound
        result["mean_normalized"] = result["normalized"].mean().item()
        result["total_normalized"] = total / bound
    return result


def compute_detection_power(
    design: torch.Tensor | np.ndarray,
    hrf_assumed: torch.Tensor | np.ndarray,
    n_conditions: int,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    tr: float = 1.0,
    device: torch.device | None = None,
    poly_degree: int = 0,
    rho: float = 0.0,
    nuisance: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Detection power under an assumed HRF, Liu & Frank (2004) Eqs. 7-11.

    Each trial type's regressor is z_i = X_i h0; the Q regressors are fitted
    jointly, whitened and detrended as in Eq. 2, and for every trial type and
    pairwise difference

        R_ij  = [D_ij (Z_perp^T Z_perp)^-1 D_ij^T]^-1 / (h0^T h0)   (Eq. 10)
        R_tot = harmonic mean of R_ij                              (Eq. 11)

    R_ij is the non-centrality of the contrast's F-test per unit squared
    amplitude, normalised by the HRF's own power so that it does not depend on
    how h0 is scaled.

    Parameters
    ----------
    design : array-like, shape (n_timepoints, n_conditions)
        Stimulus pattern per trial type on the TR grid.
    hrf_assumed : array-like, shape (k,)
        Assumed HRF h0 on the TR grid.
    n_conditions : int
        Number of trial types Q.
    effect_size : float
        Response amplitude in units of ``hrf_assumed`` (the regressor is the
        stimulus convolved with h0 as given, not a normalised copy).
    noise_std : float
        Noise standard deviation (of the innovations, when rho != 0).
    poly_degree, rho, nuisance
        As in :func:`compute_estimation_efficiency`.

    Returns
    -------
    dict with
        'per_condition' : tensor (Q,), R_ii
        'per_contrast'  : {(i, j): R_ij}
        'total'         : R_tot (Eq. 11)
        'mean'          : mean of 'per_condition'
        'snr'           : tensor (Q,), the expected t-statistic of each trial
                          type's amplitude, effect_size / SE(amplitude)
        'mean_snr'      : its mean
        'total_normalized' : R_tot / (N k / (2 (Q + 1))), the Eq. 27 bound
    """
    if device is None:
        device = torch.device("cpu")
    design = torch.as_tensor(design, device=device).to(torch.float64)
    h0 = torch.as_tensor(hrf_assumed, device=device).to(torch.float64).flatten()
    nuisance_t = None if nuisance is None else torch.as_tensor(nuisance, device=device)

    n_timepoints = design.shape[0]
    hrf_length = h0.numel()
    onsets = _condition_onsets(design, n_conditions, hrf_length)
    Z = torch.stack(
        [
            compute_design_matrix_for_condition(
                onsets, k, n_timepoints, mode="fir", hrf_length=hrf_length, device=device
            )
            @ h0
            for k in range(n_conditions)
        ],
        dim=1,
    )
    Z_perp = _whitened_detrended(Z, poly_degree, rho, nuisance_t)
    fisher = Z_perp.T @ Z_perp
    h_power = float(h0 @ h0)

    pairs = _condition_and_pairwise_contrasts(n_conditions)
    variances = _contrast_variances(fisher, [d.to(device) for _, d in pairs])
    per_contrast = {
        ij: (0.0 if np.isinf(v) else 1.0 / (v * h_power))
        for (ij, _), v in zip(pairs, variances, strict=True)
    }
    per_condition = torch.tensor([per_contrast[(k, k)] for k in range(n_conditions)])
    inv_mean = float(np.mean([1.0 / r if r > 0 else np.inf for r in per_contrast.values()]))
    total = 0.0 if np.isinf(inv_mean) else 1.0 / inv_mean

    # Var(amplitude_i) = sigma^2 / (R_ii h0^T h0), so the expected t is this.
    snr = effect_size * torch.sqrt(per_condition * h_power) / noise_std

    return {
        "per_condition": per_condition,
        "per_contrast": per_contrast,
        "total": total,
        "mean": per_condition.mean().item(),
        "snr": snr,
        "mean_snr": snr.mean().item(),
        "total_normalized": total / (n_timepoints * hrf_length / (2.0 * (n_conditions + 1))),
    }


def compute_conditional_entropy(
    onsets: torch.Tensor | np.ndarray,
    n_conditions: int,
    tr: float = 1.0,
    device: torch.device | None = None,
    order: int = 1,
) -> dict[str, Any]:
    """
    Compute conditional entropy (randomness) of design

    Conditional entropy H_r measures how unpredictable event timing is,
    given the previous event type.

    Liu & Frank (2004), Equation 14:
        H_r ≈ log₂(Q * ε_norm + 1)

    where:
        Q = hrf_length
        ε_norm = normalized efficiency

    Interpretation:
        - H_r = 0: Completely predictable (pure block design)
        - H_r = high: Highly random (m-sequence, random design)
        - Higher entropy → fewer confounds with task timing
        - BUT: trades off with power/efficiency

    'conditional_entropy' is H_r proper (Eq. 28), over the slot sequence of Q
    trial types plus null. 'type_entropy' is the same over the event-type
    sequence alone. 'total' / 'per_condition' are the entropy of the ISI
    histogram, a different quantity kept for compatibility.

    Parameters
    ----------
    onsets : array-like, shape (n_timepoints, n_conditions)
        Onset matrix (binary indicators)
    n_conditions : int
        Number of conditions
    tr : float, default=1.0
        Repetition time (for ISI calculation)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    entropy : dict with keys:
        'total': float
            Total entropy across all conditions
        'per_condition': dict
            Entropy for each condition
        'isi_distribution': dict
            ISI histogram for each condition
    """
    if device is None:
        device = torch.device("cpu")

    # Convert to tensor
    if not torch.is_tensor(onsets):
        onsets = torch.tensor(onsets, dtype=torch.float32, device=device)
    else:
        onsets = onsets.to(device)

    entropies = []
    isi_distributions = {}

    for k in range(n_conditions):
        onsets_k = onsets[:, k]
        event_times = torch.where(onsets_k > 0.5)[0]

        if len(event_times) < 2:
            # Need at least 2 events to compute ISI
            entropies.append(0.0)
            isi_distributions[k] = {}
            continue

        # Compute ISIs
        isis = []
        for i in range(len(event_times) - 1):
            isi = (event_times[i + 1] - event_times[i]).item() * tr
            isis.append(isi)

        # Compute ISI distribution (histogram)
        if len(isis) > 0:
            isis_array = np.array(isis)
            # Use bins at each TR
            min_isi = max(1, int(np.floor(isis_array.min())))
            max_isi = int(np.ceil(isis_array.max()))
            bins = np.arange(min_isi, max_isi + 2) - 0.5

            counts, _ = np.histogram(isis_array, bins=bins)
            probabilities = counts / counts.sum()

            # Compute entropy: H = -Σ p * log₂(p)
            entropy_k = 0.0
            for p in probabilities:
                if p > 0:
                    entropy_k -= p * np.log2(p)

            entropies.append(entropy_k)

            # Store ISI distribution
            isi_distributions[k] = {
                "isis": isis_array,
                "min": isis_array.min(),
                "max": isis_array.max(),
                "mean": isis_array.mean(),
                "std": isis_array.std(),
                "histogram": (counts, bins),
            }
        else:
            entropies.append(0.0)
            isi_distributions[k] = {}

    # Liu & Frank's H_r proper: how unpredictable is the NEXT TRIAL TYPE given the
    # previous r, over an alphabet of Q types plus null. The ISI entropy above is a
    # different quantity -- it is unbounded (a random design measured 1.95 bits
    # where H_r for Q=1 cannot exceed log2(2) = 1), and it is a per-condition
    # statistic, so it cannot be compared against the theoretical relation
    # H_r ~ log2(1 + Q * eps_norm) that motivates the metric. Both are reported.
    symbols = torch.zeros(onsets.shape[0], dtype=torch.long, device=device)
    for k in range(n_conditions):
        symbols[onsets[:, k] > 0.5] = k + 1  # 0 is the null/no-event symbol
    symbols_np = symbols.cpu().numpy()
    conditional = _conditional_entropy_rate(symbols_np, n_conditions + 1, order=order)

    # Two sequences, two questions, and they do not substitute for each other.
    #
    # On the slot grid above, the alphabet is Q types plus null, so the answer
    # folds in *when* an event happens: a block design scores ~0 and a p=0.5
    # random design scores the full log2(Q+1), exactly as Liu & Frank describe.
    # But when events are sparse on that grid, nearly every context is
    # null-followed-by-null, and trial-type structure washes out -- a perfectly
    # alternating ABAB and a randomly typed sequence with identical onsets both
    # measured 0.61 bits.
    #
    # So also take the entropy over the event-type sequence alone, which drops
    # timing and isolates "given the last trial's type, how surprising is the
    # next one": 0 for ABAB, 1 bit for random types over two conditions.
    event_types = symbols_np[symbols_np > 0] - 1
    type_entropy = (
        _conditional_entropy_rate(event_types, n_conditions, order=order)
        if n_conditions > 1
        else 0.0
    )

    result = {
        "total": sum(entropies),
        "mean": np.mean(entropies),
        "per_condition": {k: entropies[k] for k in range(n_conditions)},
        "isi_distribution": isi_distributions,
        "isi_entropy": sum(entropies),
        "conditional_entropy": conditional,
        "max_entropy": float(np.log2(n_conditions + 1)),
        "normalized_entropy": conditional / float(np.log2(n_conditions + 1)),
        "type_entropy": type_entropy,
        "max_type_entropy": float(np.log2(n_conditions)) if n_conditions > 1 else 0.0,
        "order": order,
    }

    return result


def _conditional_entropy_rate(symbols: np.ndarray, n_symbols: int, order: int = 1) -> float:
    """H(X_t | X_{t-1}, ..., X_{t-order}) in bits, from empirical counts.

    Zero for a fully predictable sequence, log2(n_symbols) for an i.i.d. uniform
    one. Contexts seen only once contribute zero entropy, which is the honest
    empirical answer but does bias the estimate downward when `order` is large
    relative to the sequence length.
    """
    if symbols.size <= order:
        return 0.0

    counts: dict[tuple[int, ...], np.ndarray] = {}
    for t in range(order, symbols.size):
        context = tuple(int(v) for v in symbols[t - order : t])
        if context not in counts:
            counts[context] = np.zeros(n_symbols)
        counts[context][int(symbols[t])] += 1

    total = sum(c.sum() for c in counts.values())
    entropy = 0.0
    for context_counts in counts.values():
        n_context = context_counts.sum()
        probabilities = context_counts[context_counts > 0] / n_context
        entropy += (n_context / total) * float(-(probabilities * np.log2(probabilities)).sum())
    return entropy


def compute_efficiency_power_tradeoff(
    hrf_length: int,
    n_conditions: int = 1,
    alpha_range: tuple[float, float] | None = None,
    n_points: int = 100,
    device: torch.device | None = None,
    theta_deg: float = 45.0,
) -> dict[str, Any]:
    """
    Theoretical efficiency-power trade-off, Liu et al. (2001) / Liu & Frank (2004) Eqs. 18-19.

    The eigenvalues of the stimulus autocorrelation A_k are modelled as one
    dominant eigenvalue alpha*M and k-1 equal ones (1-alpha)*M/(k-1), for
    alpha in [1/k, 1]:

        R(alpha, theta) / M = alpha cos^2(theta) + (1 - alpha) sin^2(theta) / (k - 1)
        xi(alpha) / M       = alpha (1 - alpha) / (1 + alpha (k^2 - 2k))

    alpha = 1/k spreads the eigenvalues evenly (maximum efficiency, a random or
    m-sequence design); alpha = 1 leaves one (maximum power, a block design).
    theta is the angle between h0 and the dominant eigenvector: Liu et al.
    (2001) put a 1-block design near 45 degrees and a fast 32-block design near
    90, so theta is what caps a real block design at a fraction of the bound.

    The common factor N f(p, Q) (Eqs. 20-21) scales both axes equally, so each
    curve is returned relative to its own maximum -- efficiency to its value at
    alpha = 1/k, power to its value at alpha = 1 and theta = 0. Q enters only
    through that factor and does not change the normalised curve.

    Returns
    -------
    dict with 'alpha', 'efficiency', 'power' (np.ndarray) and 'theta_deg'.
    """
    k = int(hrf_length)
    if k < 2:
        raise ValueError("hrf_length must be at least 2 for a trade-off to exist")
    lo, hi = alpha_range if alpha_range is not None else (1.0 / k, 1.0)
    alphas = np.linspace(max(lo, 1.0 / k), min(hi, 1.0), n_points)

    theta = np.deg2rad(theta_deg)
    power = alphas * np.cos(theta) ** 2 + (1.0 - alphas) * np.sin(theta) ** 2 / (k - 1)
    efficiency = alphas * (1.0 - alphas) / (1.0 + alphas * (k**2 - 2 * k))
    efficiency = efficiency / (1.0 / k**2)  # its value at alpha = 1/k

    return {
        "alpha": alphas,
        "efficiency": efficiency,
        "power": power,
        "theta_deg": theta_deg,
        "n_conditions": n_conditions,
    }


def evaluate_design(
    design: torch.Tensor | np.ndarray,
    hrf_assumed: torch.Tensor | np.ndarray,
    n_conditions: int,
    tr: float = 1.0,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    poly_degree: int = 0,
    rho: float = 0.0,
) -> dict[str, Any]:
    """
    Complete design evaluation: efficiency + power + entropy

    Convenience function that computes all three metrics.

    Parameters
    ----------
    design : array-like
        Design matrix or onset matrix
    hrf_assumed : array-like
        Assumed HRF
    n_conditions : int
        Number of conditions
    tr : float, default=1.0
        Repetition time
    effect_size : float, default=1.0
        Expected effect size
    noise_std : float, default=1.0
        Noise standard deviation
    device : torch.device, optional
        Device for computation

    Returns
    -------
    metrics : dict with keys:
        'efficiency': dict from compute_estimation_efficiency
        'power': dict from compute_detection_power
        'entropy': dict from compute_conditional_entropy
        'summary': dict with key metrics for quick comparison
    """
    if device is None:
        device = torch.device("cpu")

    # Ensure tensors
    if not torch.is_tensor(design):
        design = torch.tensor(design, dtype=torch.float32, device=device)
    if not torch.is_tensor(hrf_assumed):
        hrf_assumed = torch.tensor(hrf_assumed, dtype=torch.float32, device=device)

    hrf_length = len(hrf_assumed)

    # Compute all metrics
    efficiency = compute_estimation_efficiency(
        design,
        n_conditions,
        hrf_length,
        tr,
        normalize=True,
        device=device,
        poly_degree=poly_degree,
        rho=rho,
    )

    power = compute_detection_power(
        design,
        hrf_assumed,
        n_conditions,
        effect_size,
        noise_std,
        tr,
        device,
        poly_degree=poly_degree,
        rho=rho,
    )

    entropy = compute_conditional_entropy(design, n_conditions, tr, device)

    # Summary for quick comparison. 'entropy_total' is Liu & Frank's H_r, not
    # the summed ISI entropy it once reported -- that is a different, unbounded
    # quantity (see compute_conditional_entropy) and still lives in 'entropy'.
    summary = {
        "efficiency_mean": efficiency["mean"],
        "efficiency_total": efficiency["total"],
        "power_mean": power["mean"],
        "power_total": power["total"],
        "entropy_total": entropy["conditional_entropy"],
        "snr_mean": power["mean_snr"],
    }

    # Add efficiency-normalized if available
    if "mean_normalized" in efficiency:
        summary["efficiency_normalized"] = efficiency["mean_normalized"]

    result = {
        "efficiency": efficiency,
        "power": power,
        "entropy": entropy,
        "summary": summary,
    }

    return result


def compare_designs(
    designs_dict: dict[str, torch.Tensor | np.ndarray],
    hrf_assumed: torch.Tensor | np.ndarray,
    n_conditions: int,
    tr: float = 1.0,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Compare multiple designs on efficiency, power, entropy

    Parameters
    ----------
    designs_dict : dict
        Dictionary mapping design names to design matrices
    hrf_assumed : array-like
        Assumed HRF
    n_conditions : int
        Number of conditions
    tr : float, default=1.0
        Repetition time
    effect_size : float, default=1.0
        Expected effect size
    noise_std : float, default=1.0
        Noise standard deviation
    device : torch.device, optional
        Device for computation

    Returns
    -------
    comparison : dict
        Dictionary mapping design names to evaluation results
        Plus 'summary_table' with key metrics for all designs
    """
    if device is None:
        device = torch.device("cpu")

    results = {}
    summary_table = []

    for design_name, design in designs_dict.items():
        metrics = evaluate_design(
            design, hrf_assumed, n_conditions, tr, effect_size, noise_std, device
        )
        results[design_name] = metrics

        # Add to summary table
        summary_row = {"design": design_name}
        summary_row.update(metrics["summary"])
        summary_table.append(summary_row)

    results["summary_table"] = summary_table

    return results


def design_contrast_variance(
    design: torch.Tensor | np.ndarray,
    contrasts: torch.Tensor | np.ndarray,
    n_timepoints_per_run: list[int] | None = None,
    poly_degree: int = 2,
    arma_a: float = 0.0,
    arma_b: float = 0.0,
    extra_nuisance: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Variance of each contrast's GLS estimate, per unit noise variance.

    Works on any design matrix -- sub-TR onsets, durations, basis sets -- so it
    covers what the TR-grid Liu & Frank metrics cannot. The model is the one
    the GLM tools fit: task columns, per-run Legendre polynomials in
    block-diagonal form (glm.core.construct_polynomial_matrix), and ARMA(1,1)
    noise in AFNI's (a, b) form, block-diagonal across runs. Returns
    c (X^T R^-1 X)^-1 c^T with R the noise *correlation* matrix, so multiplying
    by the noise variance ((baseline / tSNR)^2) gives Var(c beta_hat), and

        expected t = c beta / sqrt(variance * sigma^2)

    Inestimable contrasts return inf.

    Parameters
    ----------
    design : (n_timepoints, n_columns) task regressors
    contrasts : (n_contrasts, n_columns) or (n_columns,)
    n_timepoints_per_run : run lengths; default one run
    poly_degree : per-run Legendre degree (-1 = none)
    arma_a, arma_b : AFNI ARMA(1,1) parameters (0, 0 = white);
        see simulation.noise.ou_to_arma11 for the physical parametrisation
    extra_nuisance : (n_timepoints, l) further nuisance columns

    Returns
    -------
    (n_contrasts,) float64 tensor
    """
    from fastfuncstuff.glm.arma import build_arma11_covariance
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    cpu = torch.device("cpu")
    X = torch.as_tensor(design, dtype=torch.float64, device=cpu)
    C = torch.as_tensor(contrasts, dtype=torch.float64, device=cpu)
    if C.ndim == 1:
        C = C.unsqueeze(0)
    n_t, n_task = X.shape
    if C.shape[1] != n_task:
        raise ValueError(f"contrasts have {C.shape[1]} columns, design has {n_task}")
    runs = list(n_timepoints_per_run) if n_timepoints_per_run is not None else [n_t]
    if sum(runs) != n_t:
        raise ValueError(f"run lengths sum to {sum(runs)}, design has {n_t} rows")

    blocks = [construct_polynomial_matrix(n, poly_degree, cpu, torch.float64) for n in runs]
    nuisance = torch.block_diag(*blocks) if poly_degree >= 0 else torch.empty(n_t, 0)
    if extra_nuisance is not None:
        nuisance = torch.cat(
            [nuisance, torch.as_tensor(extra_nuisance, dtype=torch.float64)], dim=1
        )
    full = torch.cat([X, nuisance], dim=1)

    if arma_a == 0.0 and arma_b == 0.0:
        whitened = full
    else:
        run_starts = np.concatenate([[0], np.cumsum(runs)[:-1]]).astype(int).tolist()
        R = build_arma11_covariance(arma_a, arma_b, n_t, cpu, torch.float64, run_starts=run_starts)
        if R is None:
            raise ValueError(f"ARMA(1,1) a={arma_a}, b={arma_b} is not a valid correlation")
        L = torch.linalg.cholesky(R)
        whitened = torch.linalg.solve_triangular(L, full, upper=False)

    fisher = whitened.T @ whitened
    L_full = torch.cat([C, torch.zeros(C.shape[0], nuisance.shape[1], dtype=torch.float64)], dim=1)
    return torch.tensor(_contrast_variances(fisher, [row[None, :] for row in L_full]))
