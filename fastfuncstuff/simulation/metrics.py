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


def _residualise(X: torch.Tensor, nuisance: torch.Tensor | None) -> torch.Tensor:
    """Project ``nuisance`` out of ``X``.

    Every real GLM carries at least a baseline, and design efficiency that
    ignores it credits the design for DC power no contrast can ever use.
    """
    if nuisance is None or nuisance.shape[1] == 0:
        return X
    q, _ = torch.linalg.qr(nuisance)
    return X - q @ (q.T @ X)


def _baseline_nuisance(
    n_timepoints: int,
    poly_degree: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
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


def compute_estimation_efficiency(
    design: torch.Tensor | np.ndarray,
    n_conditions: int,
    hrf_length: int,
    tr: float = 1.0,
    normalize: bool = True,
    device: torch.device | None = None,
    poly_degree: int = 0,
) -> torch.Tensor | dict[str, torch.Tensor | float]:
    """
    Compute estimation efficiency for HRF shape estimation

    Efficiency for condition k:
        ε_k = Tr[A_k^(-1)]^(-1)

    where A_k = (X_k^T X_k) / N
    X_k = FIR design matrix for condition k (n_timepoints x hrf_length)

    Higher efficiency = better HRF shape estimation

    Liu & Frank (2004), Equation 9:
        ε_k ≈ N * f(p, Q) / Tr[A_k^(-1)]

    where:
        N = total number of timepoints
        p = probability of event occurrence
        Q = hrf_length (number of time lags)
        f(p, Q) = p(1-p) / (1 + 2p(Q-1))

    Parameters
    ----------
    design : array-like, shape (n_timepoints, n_regressors)
        Full design matrix or onset matrix
    n_conditions : int
        Number of conditions
    hrf_length : int
        Length of HRF in TRs (Q in Liu & Frank notation)
    tr : float, default=1.0
        Repetition time (for normalization)
    normalize : bool, default=True
        If True, normalize efficiency by N*f(p,Q) to get ε_norm
    device : torch.device, optional
        Device for computation

    Returns
    -------
    efficiency : dict with keys:
        'per_condition': torch.Tensor, shape (n_conditions,)
            Efficiency for each condition
        'total': float
            Total efficiency (sum across conditions)
        'normalized': torch.Tensor, shape (n_conditions,) if normalize=True
            Normalized efficiency (divided by theoretical max)
    """
    if device is None:
        device = torch.device("cpu")

    # Convert to tensor
    if not torch.is_tensor(design):
        design = torch.tensor(design, dtype=torch.float32, device=device)
    else:
        design = design.to(device)

    n_timepoints = design.shape[0]
    nuisance = _baseline_nuisance(n_timepoints, poly_degree, device, design.dtype)

    # For FIR efficiency, we need the onset matrix (binary indicators)
    # If design is already convolved, we need to extract onsets
    # Assume design has n_conditions * hrf_length regressors (FIR style)
    # OR n_conditions regressors (onset style)

    efficiencies = []
    efficiencies_norm = []

    for k in range(n_conditions):
        # Extract onsets for condition k
        if design.shape[1] == n_conditions:
            # Design is onset matrix
            onsets_k = design[:, k]
        elif design.shape[1] == n_conditions * hrf_length:
            # Design is FIR matrix - extract first lag for each condition
            onsets_k = design[:, k * hrf_length]
        else:
            # Assume design is onset matrix with potential extra regressors
            onsets_k = (
                design[:, k] if k < design.shape[1] else torch.zeros(n_timepoints, device=device)
            )

        # Build X_k: FIR design matrix for condition k
        X_k = compute_design_matrix_for_condition(
            onsets_k[:, None], 0, n_timepoints, mode="fir", hrf_length=hrf_length, device=device
        )
        event_times = torch.where(onsets_k > 0.5)[0]

        # Efficiency is measured on what is left after the nuisance model, for
        # the same reason detection power is: a baseline the GLM will fit is not
        # available to any contrast.
        X_k = _residualise(X_k, nuisance)

        # Compute A_k = (X_k^T X_k) / N
        XtX = X_k.T @ X_k
        A_k = XtX / n_timepoints

        # Add small regularization for numerical stability
        A_k = A_k + 1e-6 * torch.eye(hrf_length, device=device)

        # Compute efficiency: ε_k = Tr[A_k^(-1)]^(-1)
        try:
            A_k_inv = torch.linalg.inv(A_k)
            trace_inv = torch.trace(A_k_inv)
            efficiency_k = 1.0 / trace_inv
        except Exception:
            # If inversion fails, efficiency is very low
            efficiency_k = torch.tensor(0.0, device=device)

        efficiencies.append(efficiency_k)

        # Normalized efficiency
        if normalize:
            # Compute f(p, Q) = p(1-p) / (1 + 2p(Q-1))
            n_events = len(event_times)
            p_k = n_events / n_timepoints  # Probability of event

            if p_k > 0:
                f_pQ = p_k * (1 - p_k) / (1 + 2 * p_k * (hrf_length - 1))
                theoretical_max = n_timepoints * f_pQ
                efficiency_norm_k = efficiency_k / theoretical_max
            else:
                efficiency_norm_k = torch.tensor(0.0, device=device)

            efficiencies_norm.append(efficiency_norm_k)

    efficiencies = torch.stack(efficiencies)

    result = {
        "per_condition": efficiencies,
        "total": efficiencies.sum().item(),
        "mean": efficiencies.mean().item(),
    }

    if normalize:
        efficiencies_norm = torch.stack(efficiencies_norm)
        result["normalized"] = efficiencies_norm
        result["mean_normalized"] = efficiencies_norm.mean().item()

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
) -> dict[str, torch.Tensor | float]:
    """
    Compute detection power for activation detection

    Power for condition k (relative efficiency):
        R_k = (h_0^T A_k h_0) / (h_0^T h_0)

    where:
        h_0 = assumed HRF (hrf_length vector)
        A_k = (X_k^T X_k) / N
        X_k = FIR design for condition k

    Higher power = better detection of activation amplitude

    Liu & Frank (2004), Equation 11:
        R_k ≈ N * f(p, Q) * (h_0^T A_k h_0) / (h_0^T h_0)

    Interpretation:
        R_k = 1 / variance_of_beta_estimate
        Higher R_k → lower variance → better detection

    Parameters
    ----------
    design : array-like, shape (n_timepoints, n_regressors)
        Design matrix or onset matrix
    hrf_assumed : array-like, shape (hrf_length,)
        Assumed HRF for detection
    n_conditions : int
        Number of conditions
    effect_size : float, default=1.0
        Expected effect size (beta)
    noise_std : float, default=1.0
        Noise standard deviation
    tr : float, default=1.0
        Repetition time
    device : torch.device, optional
        Device for computation

    Returns
    -------
    power : dict with keys:
        'per_condition': torch.Tensor, shape (n_conditions,)
            Detection power for each condition
        'total': float
            Total power (sum across conditions)
        'snr': torch.Tensor, shape (n_conditions,)
            SNR for each condition (effect_size * sqrt(power) / noise_std)
    """
    if device is None:
        device = torch.device("cpu")

    # Convert to tensors
    if not torch.is_tensor(design):
        design = torch.tensor(design, dtype=torch.float32, device=device)
    else:
        design = design.to(device)

    if not torch.is_tensor(hrf_assumed):
        hrf_assumed = torch.tensor(hrf_assumed, dtype=torch.float32, device=device)
    else:
        hrf_assumed = hrf_assumed.to(device)

    n_timepoints = design.shape[0]
    hrf_length = len(hrf_assumed)
    nuisance = _baseline_nuisance(n_timepoints, poly_degree, device, design.dtype)

    # Normalize HRF
    h0 = hrf_assumed / torch.sqrt(torch.sum(hrf_assumed**2))

    powers = []

    for k in range(n_conditions):
        # Extract onsets for condition k
        if design.shape[1] == n_conditions:
            onsets_k = design[:, k]
        elif design.shape[1] >= n_conditions * hrf_length:
            onsets_k = design[:, k * hrf_length]
        else:
            onsets_k = (
                design[:, k] if k < design.shape[1] else torch.zeros(n_timepoints, device=device)
            )

        # Build X_k (same helper the efficiency path uses)
        X_k = compute_design_matrix_for_condition(
            onsets_k[:, None], 0, n_timepoints, mode="fir", hrf_length=hrf_length, device=device
        )

        # Detection power is 1/Var(beta_hat) for the regressor X_k h0, and a real
        # GLM always fits a baseline alongside it. Leaving the baseline in counts
        # the regressor's DC against detection: measured on matched designs it was
        # 55% of a block design's score and 84% of a random one's, which
        # compressed the block-vs-random ratio from 4.5x to 1.6x and understated
        # exactly the advantage this metric exists to quantify.
        X_k = _residualise(X_k, nuisance)

        # Compute A_k = (X_k^T X_k) / N
        XtX = X_k.T @ X_k
        A_k = XtX / n_timepoints
        A_k = A_k + 1e-6 * torch.eye(hrf_length, device=device)

        # Compute power: R_k = (h_0^T A_k h_0) / (h_0^T h_0)
        # Since h0 is normalized, denominator is 1
        numerator = h0 @ A_k @ h0
        power_k = numerator  # Relative efficiency

        powers.append(power_k)

    powers = torch.stack(powers)

    # Compute SNR = effect_size * sqrt(power) / noise_std
    snr = effect_size * torch.sqrt(powers) / noise_std

    result = {
        "per_condition": powers,
        "total": powers.sum().item(),
        "mean": powers.mean().item(),
        "snr": snr,
        "mean_snr": snr.mean().item(),
    }

    return result


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

    Simplified computation (empirical):
        H_r = -Σ p(ISI) * log₂(p(ISI))

    where p(ISI) is the probability distribution of inter-stimulus intervals

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
    alpha_range: tuple[float, float] = (0.0, 1.0),
    n_points: int = 100,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute theoretical efficiency-power trade-off curve

    Liu & Frank (2004), Section 2.4:
    The trade-off is characterized by parameter α ∈ [0, 1]:
        α = 0: Maximum power, minimum efficiency
        α = 1: Maximum efficiency, minimum power

    Trade-off curves:
        ξ(α) = efficiency as function of α
        R(α) = power as function of α

    These curves define the Pareto frontier: cannot improve one without
    sacrificing the other.

    Parameters
    ----------
    hrf_length : int
        Length of HRF in TRs (Q)
    n_conditions : int, default=1
        Number of conditions
    alpha_range : tuple, default=(0.0, 1.0)
        Range of α to explore
    n_points : int, default=100
        Number of points on curve
    device : torch.device, optional
        Device for computation

    Returns
    -------
    tradeoff : dict with keys:
        'alpha': np.ndarray
            Alpha values
        'efficiency': np.ndarray
            Efficiency at each alpha
        'power': np.ndarray
            Power at each alpha
        'optimal_balanced': float
            Alpha that balances efficiency and power (α ≈ 0.5)
    """
    if device is None:
        device = torch.device("cpu")

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)

    # Theoretical relationship (simplified from Liu & Frank)
    # Efficiency: increases with α
    # Power: decreases with α

    # Based on eigenvalue distribution of A_k
    # For uniform ISI distribution:
    #   ξ(α) ∝ 1 / (1 + (1-α)²)
    #   R(α) ∝ 1 / (1 + α²)

    efficiency = 1.0 / (1.0 + (1.0 - alphas) ** 2)
    power = 1.0 / (1.0 + alphas**2)

    # Normalize to [0, 1]
    efficiency = efficiency / efficiency.max()
    power = power / power.max()

    # Find balanced point (maximize product or minimize distance from (1,1))
    balance_score = np.sqrt(efficiency**2 + power**2)
    optimal_idx = np.argmax(balance_score)

    result = {
        "alpha": alphas,
        "efficiency": efficiency,
        "power": power,
        "optimal_balanced": alphas[optimal_idx],
        "efficiency_at_optimal": efficiency[optimal_idx],
        "power_at_optimal": power[optimal_idx],
    }

    return result


def evaluate_design(
    design: torch.Tensor | np.ndarray,
    hrf_assumed: torch.Tensor | np.ndarray,
    n_conditions: int,
    tr: float = 1.0,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    device: torch.device | None = None,
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
        design, n_conditions, hrf_length, tr, normalize=True, device=device
    )

    power = compute_detection_power(
        design, hrf_assumed, n_conditions, effect_size, noise_std, tr, device
    )

    entropy = compute_conditional_entropy(design, n_conditions, tr, device)

    # Summary for quick comparison
    summary = {
        "efficiency_mean": efficiency["mean"],
        "power_mean": power["mean"],
        "entropy_total": entropy["total"],
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
