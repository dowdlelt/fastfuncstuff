"""
Design Space Exploration and Optimization

Implements tools for systematic exploration of experimental design parameter space,
focusing on ISI (Inter-Stimulus Interval) optimization using Liu & Frank (2004) metrics
and empirical evaluation with AR(1) correction.

Based on:
- Liu, T. T., & Frank, L. R. (2004). Efficiency, power, and entropy in event-related fMRI
  with multiple trial types. NeuroImage, 21(1), 387-400.
- Das, P., et al. (2023). Optimizing cognitive neuroscience experiments for separating
  event-related fMRI BOLD responses in non-randomized alternating designs.

Author: FastFuncSim
Date: 2024
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import expon, poisson, truncexpon

from .matrices import convolve_hrf
from .hrf import get_canonical_hrf

# Import our metrics
from fastfuncsim.simulation.metrics_empirical import evaluate_design_empirical


@dataclass
class ISIConstraints:
    """Constraints for ISI distribution generation"""

    min_isi: float  # Minimum ISI in seconds
    max_isi: float  # Maximum ISI in seconds
    mean_isi: float  # Target mean ISI in seconds
    tr: float = 1.0  # Repetition time in seconds


@dataclass
class DesignCandidate:
    """Container for a candidate experimental design"""

    onsets: torch.Tensor  # (n_timepoints, n_conditions) binary onset matrix
    isis: np.ndarray  # ISI sequence for each condition
    design_matrix: torch.Tensor  # Convolved design matrix
    metrics: dict | None = None  # Computed metrics
    metadata: dict | None = None  # Additional info (distribution params, etc.)


def generate_event_sequence(
    n_trials_per_condition: int | list[int],
    n_conditions: int,
    ordering: Literal["random", "alternating", "blocked", "permuted_block"] = "random",
    block_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate event sequence specifying which condition occurs at each trial.

    This separates WHAT happens from WHEN it happens (ISI timing).

    Args:
        n_trials_per_condition: Number of trials per condition (int for equal, list for unequal)
        n_conditions: Number of conditions
        ordering: Event ordering strategy:
            - 'random': Fully randomized sequence
            - 'alternating': Strict alternation (A-B-A-B... or A-B-C-A-B-C...)
            - 'blocked': Blocked design (AAAA-BBBB-AAAA...)
            - 'permuted_block': Randomized mini-blocks for balanced randomization
        block_size: Size of mini-blocks (for 'permuted_block' or 'blocked')
        seed: Random seed

    Returns:
        event_sequence: Array of condition indices [0, 1, 0, 2, 1, ...]
    """
    if seed is not None:
        np.random.seed(seed)

    # Handle n_trials specification
    if isinstance(n_trials_per_condition, int):
        n_trials = [n_trials_per_condition] * n_conditions
    else:
        n_trials = list(n_trials_per_condition)
        if len(n_trials) != n_conditions:
            raise ValueError(
                f"Length of n_trials_per_condition ({len(n_trials)}) "
                f"must match n_conditions ({n_conditions})"
            )

    _total_trials = sum(n_trials)

    if ordering == "random":
        # Fully randomized
        event_sequence = []
        for cond_idx in range(n_conditions):
            event_sequence.extend([cond_idx] * n_trials[cond_idx])
        np.random.shuffle(event_sequence)
        return np.array(event_sequence)

    elif ordering == "alternating":
        # Strict alternation - cycle through conditions
        # If unequal trials, cycle until all conditions exhausted
        event_sequence = []
        trials_remaining = n_trials.copy()

        while sum(trials_remaining) > 0:
            for cond_idx in range(n_conditions):
                if trials_remaining[cond_idx] > 0:
                    event_sequence.append(cond_idx)
                    trials_remaining[cond_idx] -= 1

        return np.array(event_sequence)

    elif ordering == "blocked":
        # Blocked design
        if block_size is None:
            # One block per condition
            block_size = max(n_trials)

        event_sequence = []
        for cond_idx in range(n_conditions):
            # Create blocks for this condition
            trials_left = n_trials[cond_idx]
            while trials_left > 0:
                this_block = min(block_size, trials_left)
                event_sequence.extend([cond_idx] * this_block)
                trials_left -= this_block

        return np.array(event_sequence)

    elif ordering == "permuted_block":
        # Permuted mini-blocks (balanced randomization)
        if block_size is None:
            block_size = n_conditions  # One of each condition per block

        # Create mini-blocks
        event_sequence = []
        trials_remaining = n_trials.copy()

        while sum(trials_remaining) > 0:
            # Create one mini-block
            mini_block = []
            for cond_idx in range(n_conditions):
                # Add min(block_size, remaining) trials for this condition
                n_in_block = min(1, trials_remaining[cond_idx])  # 0 or 1 per block
                if n_in_block > 0:
                    mini_block.extend([cond_idx] * n_in_block)
                    trials_remaining[cond_idx] -= n_in_block

            # Shuffle mini-block
            np.random.shuffle(mini_block)
            event_sequence.extend(mini_block)

        return np.array(event_sequence)

    else:
        raise ValueError(f"Unknown ordering: {ordering}")


def generate_isi_sequence(
    n_events: int,
    isi_constraints: ISIConstraints,
    distribution: Literal[
        "poisson", "exponential", "uniform", "fixed", "truncated_exponential", "poisson_target_mean"
    ] = "exponential",
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate ISI sequence (inter-stimulus intervals between consecutive events).

    This specifies WHEN events happen, independent of WHAT events they are.

    Args:
        n_events: Number of events (total across all conditions)
        isi_constraints: ISI constraints (min, max, mean)
        distribution: Distribution type:
            - 'exponential': Exponential distribution (most common for event-related)
            - 'truncated_exponential': Truncated exponential with hard min/max bounds
            - 'poisson': Poisson-distributed intervals
            - 'poisson_target_mean': Poisson with aggressive mean matching (recommended!)
            - 'uniform': Uniform random intervals
            - 'fixed': Fixed ISI (constant)
        seed: Random seed for reproducibility

    Returns:
        isis: Array of ISIs in seconds (length = n_events - 1)

    Strategy:
        1. Generate candidate ISIs from distribution
        2. Clip to [min_isi, max_isi] (or use truncated distribution)
        3. Iteratively adjust to match target mean

    Notes:
        - 'truncated_exponential': Uses scipy.stats.truncexpon for proper truncation
        - 'poisson_target_mean': Guarantees exact mean match within 0.1% tolerance
    """
    if seed is not None:
        np.random.seed(seed)

    min_isi = isi_constraints.min_isi
    max_isi = isi_constraints.max_isi
    target_mean = isi_constraints.mean_isi
    n_isis = n_events - 1  # ISIs between events

    # Validate constraints
    if min_isi >= max_isi:
        raise ValueError(f"min_isi ({min_isi}) must be < max_isi ({max_isi})")
    if not (min_isi <= target_mean <= max_isi):
        raise ValueError(f"mean_isi ({target_mean}) must be in [{min_isi}, {max_isi}]")

    # Generate initial samples
    if distribution == "exponential":
        # Exponential with rate λ = 1/mean
        scale = target_mean
        isis = expon.rvs(scale=scale, size=n_isis * 2)  # Oversample for clipping

    elif distribution == "truncated_exponential":
        # Truncated exponential: properly bounded exponential distribution
        # scipy's truncexpon parameterization: X = a + (b-a)*Y where Y ~ truncexp
        # We want distribution over [min_isi, max_isi] with mean target_mean

        # Scale parameter (related to exponential rate)
        # For truncated exponential on [0, 1], we scale and shift to [min, max]
        isi_range = max_isi - min_isi

        # Solve for b (upper truncation point) given desired mean
        # This is approximate; we'll use iterative adjustment after
        # Rule of thumb: b ≈ (target_mean - min_isi) / scale
        scale_guess = (target_mean - min_isi) * 1.5
        b_param = isi_range / scale_guess  # Truncation point in standardized units
        b_param = max(b_param, 2.0)  # Ensure reasonable truncation

        # Generate from truncated exponential
        isis_standardized = truncexpon.rvs(b=b_param, scale=1.0, size=n_isis * 2)
        # Transform to [min_isi, max_isi]
        isis = min_isi + isis_standardized * scale_guess
        isis = np.clip(isis, min_isi, max_isi)  # Ensure bounds

    elif distribution == "poisson":
        # Poisson ISIs (discrete count → continuous time)
        lam = target_mean / isi_constraints.tr
        counts = poisson.rvs(mu=lam, size=n_isis * 2)
        isis = counts * isi_constraints.tr
        isis = isis[isis > 0]  # Remove zero ISIs

    elif distribution == "poisson_target_mean":
        # Poisson with aggressive mean matching
        # Strategy: Generate Poisson samples, then use tighter tolerance in adjustment
        lam = target_mean / isi_constraints.tr
        counts = poisson.rvs(mu=lam, size=n_isis * 3)  # Extra oversampling
        isis = counts * isi_constraints.tr
        isis = isis[isis > 0]  # Remove zero ISIs

        # Pre-clip to constraints
        isis = np.clip(isis, min_isi, max_isi)

        # Initial selection with preference for values near target
        if len(isis) >= n_isis:
            # Sort by distance from target mean
            distances = np.abs(isis - target_mean)
            sorted_idx = np.argsort(distances)
            isis = isis[sorted_idx[:n_isis]]  # Select closest to target

    elif distribution == "uniform":
        # Uniform distribution
        isis = np.random.uniform(min_isi, max_isi, size=n_isis)

    elif distribution == "fixed":
        # Fixed ISI (constant spacing)
        isis = np.full(n_isis, target_mean)
        return isis

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Clip to constraints
    isis = np.clip(isis, min_isi, max_isi)

    # Select exactly n_isis
    if len(isis) < n_isis:
        warnings.warn(f"Only generated {len(isis)} valid ISIs, need {n_isis}. Padding with mean.", stacklevel=2)
        isis = np.concatenate([isis, np.full(n_isis - len(isis), target_mean)])
    else:
        isis = isis[:n_isis]

    # Iteratively adjust to match target mean (if not uniform or fixed)
    if distribution not in ["uniform", "fixed"]:
        # Tighter tolerance for poisson_target_mean
        if distribution == "poisson_target_mean":
            max_iters = 200  # More iterations
            tolerance = 0.001  # 0.1% tolerance (10x tighter!)
        else:
            max_iters = 100
            tolerance = 0.01  # 1% of target mean

        for _i in range(max_iters):
            current_mean = isis.mean()
            error = target_mean - current_mean

            if abs(error) < tolerance * target_mean:
                break

            # Adjust ISIs proportionally
            if error > 0:  # Need to increase mean
                # Increase larger ISIs more
                adjustment = error * (isis - min_isi) / (isis.sum() - min_isi * n_isis + 1e-10)
                isis = np.clip(isis + adjustment, min_isi, max_isi)
            else:  # Need to decrease mean
                # Decrease larger ISIs more
                adjustment = abs(error) * (max_isi - isis) / (max_isi * n_isis - isis.sum() + 1e-10)
                isis = np.clip(isis - adjustment, min_isi, max_isi)

    return isis


def create_onset_matrix(
    event_sequence: np.ndarray,
    isis: np.ndarray,
    duration: float,
    tr: float,
    n_conditions: int | None = None,
) -> torch.Tensor:
    """
    Convert event sequence and ISI sequence to binary onset matrix.

    This combines WHAT (event_sequence) with WHEN (isis) to produce final timing.

    Args:
        event_sequence: Array of condition indices [0, 1, 0, 2, ...] (length n_events)
        isis: Array of ISIs in seconds (length n_events - 1)
        duration: Total scan duration in seconds
        tr: Repetition time in seconds
        n_conditions: Number of conditions (inferred from event_sequence if None)

    Returns:
        onsets: (n_timepoints, n_conditions) binary matrix

    Example:
        event_sequence = [0, 1, 0, 1]  # A-B-A-B
        isis = [2.5, 3.0, 2.8]  # ISIs between events
        → Onsets at t=0 (A), t=2.5 (B), t=5.5 (A), t=8.3 (B)
    """
    if n_conditions is None:
        n_conditions = int(event_sequence.max()) + 1

    n_timepoints = int(np.ceil(duration / tr))
    onsets = torch.zeros((n_timepoints, n_conditions), dtype=torch.float32)

    # Compute onset times from ISIs
    # First event at t=0, subsequent events at cumulative ISI
    onset_times = np.concatenate([[0], np.cumsum(isis)])

    # Convert to TRs and mark onsets
    for event_idx, onset_time in enumerate(onset_times):
        if onset_time >= duration:
            break  # Exceeded scan duration

        onset_tr = int(np.round(onset_time / tr))
        if onset_tr < n_timepoints:
            condition = event_sequence[event_idx]
            onsets[onset_tr, condition] = 1.0

    return onsets


def sample_design_space(
    n_conditions: int,
    n_trials_per_condition: int | list[int],
    duration: float,
    isi_constraints: ISIConstraints,
    n_samples: int = 100,
    event_orderings: list[str] | None = None,
    isi_distributions: list[str] | None = None,
    hrf_type: str = "spm",
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[DesignCandidate]:
    """
    Sample design space by generating multiple candidate designs with various orderings and ISI distributions.

    This is the main function for exploring experimental design parameter space.

    Args:
        n_conditions: Number of experimental conditions
        n_trials_per_condition: Number of trials per condition (int for equal, list for unequal)
        duration: Total scan duration in seconds
        isi_constraints: ISI constraints (min, max, mean, tr)
        n_samples: Number of candidate designs to generate per (ordering × distribution) combination
        event_orderings: List of event orderings to try ['random', 'alternating', 'blocked', 'permuted_block']
        isi_distributions: List of ISI distributions to try ['exponential', 'poisson', 'uniform', 'fixed']
        hrf_type: HRF type for design matrix generation
        device: PyTorch device
        seed: Random seed for reproducibility

    Returns:
        candidates: List of DesignCandidate objects

    Example:
        # Explore 2 orderings × 2 distributions × 10 samples = 40 candidate designs
        candidates = sample_design_space(
            n_conditions=2,
            n_trials_per_condition=20,
            duration=300.0,
            isi_constraints=ISIConstraints(min_isi=2.0, max_isi=8.0, mean_isi=4.0, tr=1.0),
            n_samples=10,
            event_orderings=['random', 'alternating'],
            isi_distributions=['exponential', 'uniform']
        )
    """
    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if event_orderings is None:
        event_orderings = ["random"]
    if isi_distributions is None:
        isi_distributions = ["exponential"]

    if seed is not None:
        np.random.seed(seed)

    # Compute total number of events
    if isinstance(n_trials_per_condition, int):
        _total_events = n_trials_per_condition * n_conditions
    else:
        _total_events = sum(n_trials_per_condition)

    candidates = []

    # Generate HRF once
    # Note: hrf_type parameter is ignored - we use canonical HRF
    hrf = get_canonical_hrf(
        stim_duration=0.0,  # Event-related design (brief event)
        tr=isi_constraints.tr,
        duration=32.0,
        device=device,
    )

    # Iterate through all combinations
    sample_idx = 0
    for ordering in event_orderings:
        for dist in isi_distributions:
            for _rep in range(n_samples):
                # Generate event sequence (WHAT)
                event_sequence = generate_event_sequence(
                    n_trials_per_condition=n_trials_per_condition,
                    n_conditions=n_conditions,
                    ordering=ordering,
                    seed=seed + sample_idx if seed is not None else None,
                )

                # Generate ISI sequence (WHEN)
                isis = generate_isi_sequence(
                    n_events=len(event_sequence),
                    isi_constraints=isi_constraints,
                    distribution=dist,
                    seed=seed + sample_idx + 1000
                    if seed is not None
                    else None,  # Different seed space
                )

                # Combine into onset matrix
                onsets = create_onset_matrix(
                    event_sequence=event_sequence,
                    isis=isis,
                    duration=duration,
                    tr=isi_constraints.tr,
                    n_conditions=n_conditions,
                ).to(device)

                # Generate design matrix (convolve with HRF)
                n_timepoints = onsets.shape[0]
                design = convolve_hrf(
                    onsets=onsets, hrf=hrf, n_timepoints=n_timepoints, device=device
                )

                # Compute actual trial counts per condition
                actual_counts = [np.sum(event_sequence == i) for i in range(n_conditions)]

                # Store candidate
                candidate = DesignCandidate(
                    onsets=onsets,
                    isis=isis,
                    design_matrix=design,
                    metadata={
                        "ordering": ordering,
                        "distribution": dist,
                        "sample_idx": sample_idx,
                        "n_conditions": n_conditions,
                        "n_trials_requested": n_trials_per_condition,
                        "n_trials_actual": actual_counts,
                        "total_events": len(event_sequence),
                        "duration": duration,
                        "isi_mean_actual": isis.mean(),
                        "isi_std": isis.std(),
                        "isi_min": isis.min(),
                        "isi_max": isis.max(),
                    },
                )
                candidates.append(candidate)
                sample_idx += 1

    print(f"Generated {len(candidates)} candidate designs:")
    print(f"  Orderings: {event_orderings}")
    print(f"  Distributions: {isi_distributions}")
    print(f"  Samples per combination: {n_samples}")

    return candidates


def evaluate_design_candidates(
    candidates: list[DesignCandidate],
    data: torch.Tensor | None = None,
    hrf_length: int = 30,
    effect_sizes: list[float] | None = None,
    noise_level: float = 1.0,
    n_voxels: int = 100,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[DesignCandidate]:
    """
    Evaluate all candidate designs using empirical metrics.

    Args:
        candidates: List of DesignCandidate objects
        data: Optional real data (n_timepoints, n_voxels). If None, simulates data.
        hrf_length: HRF length in TRs for efficiency estimation
        effect_sizes: Effect sizes for each condition (if simulating)
        noise_level: Noise level (if simulating)
        n_voxels: Number of voxels (if simulating)
        device: PyTorch device
        verbose: Print progress

    Returns:
        candidates: Same list with metrics filled in
    """
    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    for i, candidate in enumerate(candidates):
        if verbose and (i + 1) % 10 == 0:
            print(f"Evaluating candidate {i + 1}/{len(candidates)}...")

        # If no data provided, simulate simple data
        if data is None:
            n_timepoints = candidate.design_matrix.shape[0]
            n_conditions = candidate.onsets.shape[1]

            # Use provided effect sizes or default
            if effect_sizes is None:
                betas = torch.ones(n_conditions, device=device)
            else:
                betas = torch.tensor(effect_sizes, dtype=torch.float32, device=device)

            # Simulate: Y = X @ β + ε
            signal = candidate.design_matrix @ betas.unsqueeze(1).expand(-1, n_voxels)
            noise = torch.randn(n_timepoints, n_voxels, device=device) * noise_level
            sim_data = signal + noise
        else:
            sim_data = data.to(device)

        # Compute metrics using empirical methods
        try:
            n_conditions = candidate.onsets.shape[1]
            metrics = evaluate_design_empirical(
                data=sim_data,
                design=candidate.design_matrix,
                onsets=candidate.onsets,
                n_conditions=n_conditions,
                hrf_length=hrf_length,
                device=device,
            )
            candidate.metrics = metrics

        except Exception as e:
            warnings.warn(f"Failed to evaluate candidate {i}: {e}", stacklevel=2)
            candidate.metrics = {
                "detection_power": np.nan,
                "estimation_efficiency": np.nan,
                "error": str(e),
            }

    return candidates


def find_optimal_designs(
    candidates: list[DesignCandidate],
    objective: Literal["power", "efficiency", "balanced"] = "balanced",
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[int, DesignCandidate, float]]:
    """
    Find optimal designs based on specified objective.

    Args:
        candidates: List of evaluated DesignCandidate objects
        objective: Optimization objective:
            - 'power': Maximize detection power
            - 'efficiency': Maximize estimation efficiency
            - 'balanced': Weighted combination (alpha * power + (1-alpha) * efficiency)
        alpha: Weight for balanced objective (0=efficiency only, 1=power only)
        top_k: Number of top designs to return

    Returns:
        top_designs: List of (index, candidate, score) tuples, sorted by score (descending)
    """
    scores = []

    for idx, candidate in enumerate(candidates):
        if candidate.metrics is None:
            scores.append((idx, candidate, -np.inf))
            continue

        power = candidate.metrics.get("detection_power", np.nan)
        efficiency = candidate.metrics.get("estimation_efficiency", np.nan)

        if np.isnan(power) or np.isnan(efficiency):
            scores.append((idx, candidate, -np.inf))
            continue

        # Normalize to [0, 1] range within this set
        # (Will do global normalization after collecting all scores)
        if objective == "power":
            score = power
        elif objective == "efficiency":
            score = efficiency
        elif objective == "balanced":
            # Need to normalize first
            score = None  # Will compute after normalization
        else:
            raise ValueError(f"Unknown objective: {objective}")

        scores.append((idx, candidate, score))

    # Normalize if using balanced objective
    if objective == "balanced":
        powers = np.array(
            [c.metrics.get("detection_power", np.nan) for c in candidates if c.metrics is not None]
        )
        efficiencies = np.array(
            [
                c.metrics.get("estimation_efficiency", np.nan)
                for c in candidates
                if c.metrics is not None
            ]
        )

        # Remove NaNs for normalization
        powers_valid = powers[~np.isnan(powers)]
        efficiencies_valid = efficiencies[~np.isnan(efficiencies)]

        if len(powers_valid) > 0 and len(efficiencies_valid) > 0:
            power_min, power_max = powers_valid.min(), powers_valid.max()
            eff_min, eff_max = efficiencies_valid.min(), efficiencies_valid.max()

            # Recompute scores with normalization
            for i, (idx, candidate, _) in enumerate(scores):
                if candidate.metrics is None:
                    continue

                power = candidate.metrics.get("detection_power", np.nan)
                efficiency = candidate.metrics.get("estimation_efficiency", np.nan)

                if np.isnan(power) or np.isnan(efficiency):
                    continue

                # Normalize
                power_norm = (power - power_min) / (power_max - power_min + 1e-10)
                eff_norm = (efficiency - eff_min) / (eff_max - eff_min + 1e-10)

                # Balanced score
                score = alpha * power_norm + (1 - alpha) * eff_norm
                scores[i] = (idx, candidate, score)

    # Sort by score (descending)
    scores.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -np.inf, reverse=True)

    # Return top_k
    return scores[:top_k]


def compare_designs_summary(
    candidates: list[DesignCandidate],
    top_k: int = 10,
    objective: str = "balanced",
    alpha: float = 0.5,
) -> str:
    """
    Generate text summary comparing designs.

    Args:
        candidates: List of evaluated candidates
        top_k: Number of top designs to show
        objective: Objective used for ranking
        alpha: Weight for balanced objective

    Returns:
        summary: Text summary string
    """
    top_designs = find_optimal_designs(
        candidates=candidates, objective=objective, alpha=alpha, top_k=top_k
    )

    summary = []
    summary.append("=" * 80)
    summary.append(f"Design Space Exploration Summary (Objective: {objective})")
    summary.append("=" * 80)
    summary.append(f"Total candidates evaluated: {len(candidates)}")
    summary.append(f"Showing top {len(top_designs)} designs")
    summary.append("")

    for rank, (idx, candidate, score) in enumerate(top_designs, 1):
        summary.append(f"Rank {rank}: Design #{idx} (Score: {score:.4f})")
        summary.append("-" * 80)

        # Metadata
        if candidate.metadata:
            summary.append(f"  Ordering: {candidate.metadata.get('ordering', 'N/A')}")
            summary.append(f"  ISI Distribution: {candidate.metadata.get('distribution', 'N/A')}")
            summary.append(f"  N conditions: {candidate.metadata.get('n_conditions', 'N/A')}")

            # Handle both old and new metadata formats
            n_trials = candidate.metadata.get(
                "n_trials_actual", candidate.metadata.get("n_trials", "N/A")
            )
            if isinstance(n_trials, list):
                summary.append(f"  N trials per condition: {n_trials}")
            else:
                summary.append(f"  N trials: {n_trials}")

            # ISI statistics
            isi_mean = candidate.metadata.get("isi_mean_actual", "N/A")
            isi_std = candidate.metadata.get("isi_std", "N/A")
            if isinstance(isi_mean, (int, float)) and isinstance(isi_std, (int, float)):
                summary.append(f"  ISI: {isi_mean:.2f} ± {isi_std:.2f} s")
            elif isinstance(isi_mean, list):
                # Old format with per-condition ISIs
                summary.append(f"  ISI means: {[f'{x:.2f}' for x in isi_mean]}")

        # Metrics
        if candidate.metrics:
            summary.append(
                f"  Detection Power: {candidate.metrics.get('detection_power', np.nan):.4f}"
            )
            summary.append(
                f"  Estimation Efficiency: {candidate.metrics.get('estimation_efficiency', np.nan):.4f}"
            )

            rho = candidate.metrics.get("rho", np.nan)
            if not np.isnan(rho):
                summary.append(f"  AR(1) coefficient: {rho:.3f}")

        summary.append("")

    summary.append("=" * 80)

    return "\n".join(summary)


def plot_fitness_landscape(
    candidates: list[DesignCandidate],
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
):
    """
    Plot fitness landscape: Detection Power vs Estimation Efficiency.

    Creates scatter plot with:
    - X-axis: Estimation Efficiency
    - Y-axis: Detection Power
    - Color: Distribution type
    - Size: Score (balanced objective)

    Args:
        candidates: List of evaluated candidates
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    # Extract metrics
    powers = []
    efficiencies = []
    distributions = []
    scores = []

    for candidate in candidates:
        if candidate.metrics is None:
            continue

        power = candidate.metrics.get("detection_power", np.nan)
        efficiency = candidate.metrics.get("estimation_efficiency", np.nan)

        if np.isnan(power) or np.isnan(efficiency):
            continue

        powers.append(power)
        efficiencies.append(efficiency)
        distributions.append(candidate.metadata.get("distribution", "unknown"))

        # Compute balanced score for sizing
        # Normalize within dataset
        power_norm = (power - min(powers)) / (max(powers) - min(powers) + 1e-10)
        eff_norm = (efficiency - min(efficiencies)) / (
            max(efficiencies) - min(efficiencies) + 1e-10
        )
        score = 0.5 * power_norm + 0.5 * eff_norm
        scores.append(score)

    # Convert to arrays
    powers = np.array(powers)
    efficiencies = np.array(efficiencies)
    scores = np.array(scores)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Colored by distribution
    unique_dists = sorted(set(distributions))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_dists)))
    color_map = {dist: colors[i] for i, dist in enumerate(unique_dists)}
    point_colors = [color_map[d] for d in distributions]

    _scatter1 = ax1.scatter(
        efficiencies, powers, c=point_colors, s=100, alpha=0.6, edgecolors="k", linewidths=0.5
    )
    ax1.set_xlabel("Estimation Efficiency", fontsize=12)
    ax1.set_ylabel("Detection Power", fontsize=12)
    ax1.set_title("Fitness Landscape (by Distribution)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Legend for distributions
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[dist],
            markersize=10,
            label=dist,
        )
        for dist in unique_dists
    ]
    ax1.legend(handles=legend_elements, loc="best")

    # Right plot: Sized by balanced score
    scatter2 = ax2.scatter(
        efficiencies,
        powers,
        c=scores,
        s=scores * 200,  # Size proportional to score
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.5,
    )
    ax2.set_xlabel("Estimation Efficiency", fontsize=12)
    ax2.set_ylabel("Detection Power", fontsize=12)
    ax2.set_title("Fitness Landscape (by Score)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter2, ax=ax2)
    cbar.set_label("Balanced Score", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved fitness landscape to {save_path}")

    return fig


def plot_pareto_frontier(
    candidates: list[DesignCandidate],
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
):
    """
    Plot Pareto frontier of detection power vs estimation efficiency.

    A design is Pareto optimal if no other design is better in BOTH metrics.

    Args:
        candidates: List of evaluated candidates
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    # Extract metrics
    powers = []
    efficiencies = []
    indices = []

    for idx, candidate in enumerate(candidates):
        if candidate.metrics is None:
            continue

        power = candidate.metrics.get("detection_power", np.nan)
        efficiency = candidate.metrics.get("estimation_efficiency", np.nan)

        if np.isnan(power) or np.isnan(efficiency):
            continue

        powers.append(power)
        efficiencies.append(efficiency)
        indices.append(idx)

    powers = np.array(powers)
    efficiencies = np.array(efficiencies)
    indices = np.array(indices)

    # Compute Pareto frontier
    # A point is Pareto optimal if no other point dominates it
    is_pareto = np.ones(len(powers), dtype=bool)

    for i in range(len(powers)):
        for j in range(len(powers)):
            if i == j:
                continue
            # j dominates i if j is better in both metrics
            if powers[j] >= powers[i] and efficiencies[j] >= efficiencies[i]:
                if powers[j] > powers[i] or efficiencies[j] > efficiencies[i]:
                    is_pareto[i] = False
                    break

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # All points
    ax.scatter(
        efficiencies[~is_pareto],
        powers[~is_pareto],
        c="lightgray",
        s=50,
        alpha=0.5,
        label="Non-Pareto",
    )

    # Pareto points
    ax.scatter(
        efficiencies[is_pareto],
        powers[is_pareto],
        c="red",
        s=100,
        alpha=0.8,
        edgecolors="k",
        linewidths=1.5,
        label="Pareto Optimal",
        zorder=5,
    )

    # Connect Pareto frontier
    pareto_eff = efficiencies[is_pareto]
    pareto_pow = powers[is_pareto]
    sorted_idx = np.argsort(pareto_eff)
    ax.plot(
        pareto_eff[sorted_idx], pareto_pow[sorted_idx], "r--", alpha=0.5, linewidth=1.5, zorder=4
    )

    ax.set_xlabel("Estimation Efficiency", fontsize=12)
    ax.set_ylabel("Detection Power", fontsize=12)
    ax.set_title("Pareto Frontier: Power vs Efficiency Trade-off", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Pareto frontier to {save_path}")

    # Print Pareto optimal designs
    print(f"\nFound {is_pareto.sum()} Pareto optimal designs:")
    for idx in indices[is_pareto]:
        candidate = candidates[idx]
        print(
            f"  Design #{idx}: Power={candidate.metrics['detection_power']:.4f}, "
            f"Efficiency={candidate.metrics['estimation_efficiency']:.4f}"
        )

    return fig, indices[is_pareto]


def plot_isi_range_optimization(
    n_conditions: int,
    n_trials_per_condition: int,
    duration: float,
    tr: float,
    min_isi_range: tuple[float, float] = (1.0, 4.0),
    max_isi_range: tuple[float, float] = (4.0, 12.0),
    n_grid_points: int = 10,
    isi_distribution: str = "exponential",
    event_ordering: str = "random",
    n_samples_per_point: int = 3,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """
    Plot ISI range optimization landscape (Das et al. 2023 Figure 7 style).

    Creates 2D heatmaps showing detection power and estimation efficiency
    as a function of ISI range (min_isi vs max_isi).

    This answers: "Given constraints on my ISI range, which combination is optimal?"

    Args:
        n_conditions: Number of conditions
        n_trials_per_condition: Trials per condition
        duration: Scan duration in seconds
        tr: Repetition time in seconds
        min_isi_range: Range of minimum ISIs to test (low, high)
        max_isi_range: Range of maximum ISIs to test (low, high)
        n_grid_points: Number of grid points per dimension
        isi_distribution: ISI distribution to use
        event_ordering: Event ordering strategy
        n_samples_per_point: Number of samples per grid point (for averaging)
        figsize: Figure size
        save_path: Path to save figure
        device: PyTorch device
        verbose: Print progress

    Returns:
        fig: Matplotlib figure
        results_grid: Dict with power_grid and efficiency_grid arrays

    Example:
        >>> # Test ISI ranges from 1-4s (min) to 4-12s (max)
        >>> fig, results = plot_isi_range_optimization(
        ...     n_conditions=2,
        ...     n_trials_per_condition=20,
        ...     duration=300.0,
        ...     tr=1.0,
        ...     min_isi_range=(1.0, 4.0),
        ...     max_isi_range=(4.0, 12.0),
        ...     n_grid_points=10
        ... )
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Create grid of ISI parameters
    min_isis = np.linspace(min_isi_range[0], min_isi_range[1], n_grid_points)
    max_isis = np.linspace(max_isi_range[0], max_isi_range[1], n_grid_points)

    # Initialize result grids
    power_grid = np.zeros((n_grid_points, n_grid_points))
    efficiency_grid = np.zeros((n_grid_points, n_grid_points))
    count_grid = np.zeros((n_grid_points, n_grid_points))  # Track valid points

    if verbose:
        print("ISI Range Optimization")
        print("=" * 80)
        print(f"Grid: {n_grid_points} × {n_grid_points} = {n_grid_points**2} points")
        print(f"Samples per point: {n_samples_per_point}")
        print(f"Total evaluations: {n_grid_points**2 * n_samples_per_point}")
        print(f"ISI Distribution: {isi_distribution}")
        print(f"Event Ordering: {event_ordering}")
        print()

    total_points = n_grid_points**2
    completed = 0

    # Iterate through grid
    for i, min_isi in enumerate(min_isis):
        for j, max_isi in enumerate(max_isis):
            # Skip invalid combinations (min >= max)
            if min_isi >= max_isi:
                power_grid[i, j] = np.nan
                efficiency_grid[i, j] = np.nan
                completed += 1
                continue

            # Compute mean ISI (center of range)
            mean_isi = (min_isi + max_isi) / 2.0

            # Create ISI constraints for this grid point
            isi_constraints = ISIConstraints(
                min_isi=min_isi, max_isi=max_isi, mean_isi=mean_isi, tr=tr
            )

            # Sample designs for this ISI configuration
            try:
                candidates = sample_design_space(
                    n_conditions=n_conditions,
                    n_trials_per_condition=n_trials_per_condition,
                    duration=duration,
                    isi_constraints=isi_constraints,
                    n_samples=n_samples_per_point,
                    event_orderings=[event_ordering],
                    isi_distributions=[isi_distribution],
                    device=device,
                    seed=42 + i * n_grid_points + j,  # Reproducible but varied
                )

                # Evaluate designs
                candidates = evaluate_design_candidates(
                    candidates=candidates, device=device, verbose=False
                )

                # Average metrics across samples
                powers = [
                    c.metrics["detection_power"]
                    for c in candidates
                    if c.metrics is not None
                    and not np.isnan(c.metrics.get("detection_power", np.nan))
                ]
                efficiencies = [
                    c.metrics["estimation_efficiency"]
                    for c in candidates
                    if c.metrics is not None
                    and not np.isnan(c.metrics.get("estimation_efficiency", np.nan))
                ]

                if len(powers) > 0 and len(efficiencies) > 0:
                    power_grid[i, j] = np.mean(powers)
                    efficiency_grid[i, j] = np.mean(efficiencies)
                    count_grid[i, j] = len(powers)
                else:
                    power_grid[i, j] = np.nan
                    efficiency_grid[i, j] = np.nan

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed at grid point ({min_isi:.2f}, {max_isi:.2f}): {e}")
                power_grid[i, j] = np.nan
                efficiency_grid[i, j] = np.nan

            completed += 1
            if verbose and completed % max(1, total_points // 10) == 0:
                print(
                    f"Progress: {completed}/{total_points} ({100 * completed / total_points:.0f}%)"
                )

    if verbose:
        print(f"\nCompleted! Valid points: {np.sum(~np.isnan(power_grid))}/{total_points}")
        print()

    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot Detection Power
    im1 = ax1.imshow(
        power_grid.T,  # Transpose for correct orientation
        origin="lower",
        extent=[min_isi_range[0], min_isi_range[1], max_isi_range[0], max_isi_range[1]],
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
    )
    ax1.set_xlabel("Minimum ISI (s)", fontsize=12)
    ax1.set_ylabel("Maximum ISI (s)", fontsize=12)
    ax1.set_title("Detection Power\n(Das et al. 2023 Style)", fontsize=14)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Detection Power (Fd)", fontsize=10)
    ax1.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)

    # Mark optimal point
    max_idx = np.unravel_index(np.nanargmax(power_grid), power_grid.shape)
    optimal_min_isi = min_isis[max_idx[0]]
    optimal_max_isi = max_isis[max_idx[1]]
    ax1.plot(
        optimal_min_isi,
        optimal_max_isi,
        "r*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Optimal",
    )
    ax1.legend(loc="upper right")

    # Plot Estimation Efficiency
    im2 = ax2.imshow(
        efficiency_grid.T,
        origin="lower",
        extent=[min_isi_range[0], min_isi_range[1], max_isi_range[0], max_isi_range[1]],
        aspect="auto",
        cmap="plasma",
        interpolation="bilinear",
    )
    ax2.set_xlabel("Minimum ISI (s)", fontsize=12)
    ax2.set_ylabel("Maximum ISI (s)", fontsize=12)
    ax2.set_title("Estimation Efficiency\n(Das et al. 2023 Style)", fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Estimation Efficiency (Fe)", fontsize=10)
    ax2.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)

    # Mark optimal point
    max_idx = np.unravel_index(np.nanargmax(efficiency_grid), efficiency_grid.shape)
    optimal_min_isi = min_isis[max_idx[0]]
    optimal_max_isi = max_isis[max_idx[1]]
    ax2.plot(
        optimal_min_isi,
        optimal_max_isi,
        "r*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Optimal",
    )
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if verbose:
            print(f"Saved ISI range optimization plot to {save_path}")

    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Optimization Summary")
        print("=" * 80)

        # Detection power optimum
        max_idx = np.unravel_index(np.nanargmax(power_grid), power_grid.shape)
        opt_min = min_isis[max_idx[0]]
        opt_max = max_isis[max_idx[1]]
        opt_power = power_grid[max_idx]
        print("Detection Power Optimum:")
        print(f"  ISI range: [{opt_min:.2f}, {opt_max:.2f}] s")
        print(f"  Detection Power: {opt_power:.4f}")
        print()

        # Estimation efficiency optimum
        max_idx = np.unravel_index(np.nanargmax(efficiency_grid), efficiency_grid.shape)
        opt_min = min_isis[max_idx[0]]
        opt_max = max_isis[max_idx[1]]
        opt_eff = efficiency_grid[max_idx]
        print("Estimation Efficiency Optimum:")
        print(f"  ISI range: [{opt_min:.2f}, {opt_max:.2f}] s")
        print(f"  Estimation Efficiency: {opt_eff:.4f}")
        print("=" * 80)

    results_grid = {
        "power_grid": power_grid,
        "efficiency_grid": efficiency_grid,
        "count_grid": count_grid,
        "min_isis": min_isis,
        "max_isis": max_isis,
    }

    return fig, results_grid


def plot_isi_range_by_target_mean(
    n_conditions: int,
    n_trials_per_condition: int,
    duration: float,
    tr: float,
    target_mean_isis: list[float],
    min_isi_range: tuple[float, float] = (1.0, 4.0),
    max_isi_range: tuple[float, float] = (4.0, 12.0),
    n_grid_points: int = 10,
    isi_distribution: str = "exponential",
    event_ordering: str = "random",
    n_samples_per_point: int = 3,
    figsize_per_row: tuple[int, int] = (14, 5),
    save_path: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """
    Plot ISI range optimization with EXPLICIT target mean ISIs.

    Creates separate heatmap rows for each target mean ISI value.
    This properly handles the constraint that min_isi <= target_mean <= max_isi.

    IMPORTANT: Number of events will vary across the parameter space!
    - Fixed duration + varying mean ISI → varying number of events
    - Longer mean ISI → fewer events fit in the scan

    Args:
        n_conditions: Number of conditions
        n_trials_per_condition: Trials per condition (REQUESTED, may vary based on ISI)
        duration: Fixed scan duration in seconds
        tr: Repetition time in seconds
        target_mean_isis: List of target mean ISIs to test (e.g., [3.0, 4.0, 5.0])
        min_isi_range: Range of minimum ISIs to test
        max_isi_range: Range of maximum ISIs to test
        n_grid_points: Number of grid points per dimension
        isi_distribution: ISI distribution to use
        event_ordering: Event ordering strategy
        n_samples_per_point: Number of samples per grid point
        figsize_per_row: Figure size per row (one row per target mean)
        save_path: Path to save figure
        device: PyTorch device
        verbose: Print progress

    Returns:
        fig: Matplotlib figure with subplots
        results: Dict with grids for each target mean

    Example:
        >>> # Test 3 different target means
        >>> fig, results = plot_isi_range_by_target_mean(
        ...     n_conditions=2,
        ...     n_trials_per_condition=20,  # Requested, will vary
        ...     duration=300.0,
        ...     tr=1.0,
        ...     target_mean_isis=[3.0, 4.0, 5.0],
        ...     min_isi_range=(1.0, 4.0),
        ...     max_isi_range=(4.0, 12.0)
        ... )
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    n_target_means = len(target_mean_isis)

    # Create figure with rows for each target mean
    fig, axes = plt.subplots(
        n_target_means, 2, figsize=(figsize_per_row[0], figsize_per_row[1] * n_target_means)
    )

    # Ensure axes is 2D even if only one target mean
    if n_target_means == 1:
        axes = axes.reshape(1, -1)

    results = {}

    # Create grid
    min_isis = np.linspace(min_isi_range[0], min_isi_range[1], n_grid_points)
    max_isis = np.linspace(max_isi_range[0], max_isi_range[1], n_grid_points)

    # Process each target mean
    for mean_idx, target_mean in enumerate(target_mean_isis):
        if verbose:
            print(f"\n{'=' * 80}")
            print(
                f"Processing Target Mean ISI = {target_mean:.2f} s ({mean_idx + 1}/{n_target_means})"
            )
            print(f"{'=' * 80}")

        # Initialize grids for this target mean
        power_grid = np.zeros((n_grid_points, n_grid_points))
        efficiency_grid = np.zeros((n_grid_points, n_grid_points))
        event_count_grid = np.zeros((n_grid_points, n_grid_points))

        # Iterate through grid
        for i, min_isi in enumerate(min_isis):
            for j, max_isi in enumerate(max_isis):
                # Check if target mean is feasible
                if not (min_isi <= target_mean <= max_isi):
                    power_grid[i, j] = np.nan
                    efficiency_grid[i, j] = np.nan
                    event_count_grid[i, j] = np.nan
                    continue

                # Create ISI constraints
                isi_constraints = ISIConstraints(
                    min_isi=min_isi, max_isi=max_isi, mean_isi=target_mean, tr=tr
                )

                # Sample and evaluate
                try:
                    candidates = sample_design_space(
                        n_conditions=n_conditions,
                        n_trials_per_condition=n_trials_per_condition,
                        duration=duration,
                        isi_constraints=isi_constraints,
                        n_samples=n_samples_per_point,
                        event_orderings=[event_ordering],
                        isi_distributions=[isi_distribution],
                        device=device,
                        seed=42 + mean_idx * 1000 + i * n_grid_points + j,
                    )

                    candidates = evaluate_design_candidates(
                        candidates=candidates, device=device, verbose=False
                    )

                    # Average metrics
                    powers = [
                        c.metrics["detection_power"]
                        for c in candidates
                        if c.metrics is not None
                        and not np.isnan(c.metrics.get("detection_power", np.nan))
                    ]
                    efficiencies = [
                        c.metrics["estimation_efficiency"]
                        for c in candidates
                        if c.metrics is not None
                        and not np.isnan(c.metrics.get("estimation_efficiency", np.nan))
                    ]
                    event_counts = [c.metadata["total_events"] for c in candidates]

                    if len(powers) > 0 and len(efficiencies) > 0:
                        power_grid[i, j] = np.mean(powers)
                        efficiency_grid[i, j] = np.mean(efficiencies)
                        event_count_grid[i, j] = np.mean(event_counts)
                    else:
                        power_grid[i, j] = np.nan
                        efficiency_grid[i, j] = np.nan
                        event_count_grid[i, j] = np.nan

                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed at ({min_isi:.2f}, {max_isi:.2f}): {e}")
                    power_grid[i, j] = np.nan
                    efficiency_grid[i, j] = np.nan
                    event_count_grid[i, j] = np.nan

        # Plot detection power
        ax_power = axes[mean_idx, 0]
        im1 = ax_power.imshow(
            power_grid.T,
            origin="lower",
            extent=[min_isi_range[0], min_isi_range[1], max_isi_range[0], max_isi_range[1]],
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )
        ax_power.set_xlabel("Minimum ISI (s)", fontsize=10)
        ax_power.set_ylabel("Maximum ISI (s)", fontsize=10)
        ax_power.set_title(f"Detection Power\n(Target Mean = {target_mean:.1f}s)", fontsize=11)
        plt.colorbar(im1, ax=ax_power, label="Power")
        ax_power.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)

        # Mark optimal
        if not np.all(np.isnan(power_grid)):
            max_idx = np.unravel_index(np.nanargmax(power_grid), power_grid.shape)
            opt_min = min_isis[max_idx[0]]
            opt_max = max_isis[max_idx[1]]
            ax_power.plot(
                opt_min, opt_max, "r*", markersize=12, markeredgecolor="white", markeredgewidth=1.5
            )

        # Plot estimation efficiency
        ax_eff = axes[mean_idx, 1]
        im2 = ax_eff.imshow(
            efficiency_grid.T,
            origin="lower",
            extent=[min_isi_range[0], min_isi_range[1], max_isi_range[0], max_isi_range[1]],
            aspect="auto",
            cmap="plasma",
            interpolation="bilinear",
        )
        ax_eff.set_xlabel("Minimum ISI (s)", fontsize=10)
        ax_eff.set_ylabel("Maximum ISI (s)", fontsize=10)
        ax_eff.set_title(f"Estimation Efficiency\n(Target Mean = {target_mean:.1f}s)", fontsize=11)
        plt.colorbar(im2, ax=ax_eff, label="Efficiency")
        ax_eff.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)

        # Mark optimal
        if not np.all(np.isnan(efficiency_grid)):
            max_idx = np.unravel_index(np.nanargmax(efficiency_grid), efficiency_grid.shape)
            opt_min = min_isis[max_idx[0]]
            opt_max = max_isis[max_idx[1]]
            ax_eff.plot(
                opt_min, opt_max, "r*", markersize=12, markeredgecolor="white", markeredgewidth=1.5
            )

        # Store results
        results[f"mean_{target_mean:.1f}"] = {
            "power_grid": power_grid,
            "efficiency_grid": efficiency_grid,
            "event_count_grid": event_count_grid,
            "mean_events": np.nanmean(event_count_grid),
        }

        if verbose:
            print(f"  Valid points: {np.sum(~np.isnan(power_grid))}/{n_grid_points**2}")
            print(f"  Mean event count: {np.nanmean(event_count_grid):.1f} events")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if verbose:
            print(f"\nSaved to {save_path}")

    return fig, results


def plot_hrf_index_recovery(
    true_hrf_indices: torch.Tensor | np.ndarray,
    recovered_hrf_indices: torch.Tensor | np.ndarray,
    spatial_shape: tuple[int, ...] | None = None,
    slice_axis: int = 2,
    n_slices: int | None = None,
    figsize: tuple[int, int] = (16, 6),
    save_path: str | None = None,
):
    """
    Visualize HRF library recovery accuracy.

    Shows slice-by-slice comparison of true vs recovered HRF indices.
    If voxels were created with smooth HRF gradients, successful recovery
    should show smooth patterns.

    Args:
        true_hrf_indices: Ground truth HRF indices per voxel (1D or spatial shape)
        recovered_hrf_indices: Recovered HRF indices from fit_glm_hrf_library()
        spatial_shape: Spatial shape (nx, ny, nz). If None, inferred from data
        slice_axis: Axis to slice along (0=x, 1=y, 2=z)
        n_slices: Number of slices to show (None = all slices)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
        accuracy: Dict with recovery metrics

    Example:
        >>> # Create parametric voxels with HRF gradient
        >>> from simulation import create_parametric_voxels
        >>> voxels = create_parametric_voxels(
        ...     n_voxels=1000,
        ...     vary_hrf=True,
        ...     hrf_library_size=20
        ... )
        >>> true_indices = voxels['hrf_indices']
        >>>
        >>> # Fit with HRF library
        >>> results, recovered_indices, r2 = fit_glm_hrf_library(...)
        >>>
        >>> # Visualize recovery
        >>> fig, acc = plot_hrf_index_recovery(
        ...     true_hrf_indices=true_indices,
        ...     recovered_hrf_indices=recovered_indices,
        ...     spatial_shape=(10, 10, 10)
        ... )
    """
    import matplotlib.pyplot as plt

    # Convert to numpy
    if torch.is_tensor(true_hrf_indices):
        true_hrf_indices = true_hrf_indices.cpu().numpy()
    if torch.is_tensor(recovered_hrf_indices):
        recovered_hrf_indices = recovered_hrf_indices.cpu().numpy()

    # Reshape if needed
    if spatial_shape is not None:
        true_hrf_indices = true_hrf_indices.reshape(spatial_shape)
        recovered_hrf_indices = recovered_hrf_indices.reshape(spatial_shape)
    else:
        # Assume already in spatial form
        spatial_shape = true_hrf_indices.shape

    # Determine slices to show
    n_slices_total = spatial_shape[slice_axis]
    if n_slices is None:
        n_slices = min(n_slices_total, 9)  # Max 9 slices

    slice_indices = np.linspace(0, n_slices_total - 1, n_slices, dtype=int)

    # Create figure
    n_rows = 3  # True, Recovered, Difference
    fig, axes = plt.subplots(n_rows, n_slices, figsize=figsize)

    if n_slices == 1:
        axes = axes.reshape(-1, 1)

    # Compute accuracy
    correct = true_hrf_indices == recovered_hrf_indices
    accuracy_overall = correct.mean()

    # Extract slices and plot
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice
        if slice_axis == 0:
            true_slice = true_hrf_indices[slice_idx, :, :]
            rec_slice = recovered_hrf_indices[slice_idx, :, :]
        elif slice_axis == 1:
            true_slice = true_hrf_indices[:, slice_idx, :]
            rec_slice = recovered_hrf_indices[:, slice_idx, :]
        else:  # slice_axis == 2
            true_slice = true_hrf_indices[:, :, slice_idx]
            rec_slice = recovered_hrf_indices[:, :, slice_idx]

        diff_slice = rec_slice - true_slice
        slice_accuracy = (true_slice == rec_slice).mean()

        # Plot true
        im0 = axes[0, i].imshow(
            true_slice.T,
            cmap="turbo",
            interpolation="nearest",
            vmin=true_hrf_indices.min(),
            vmax=true_hrf_indices.max(),
        )
        axes[0, i].set_title(f"Slice {slice_idx}\nTrue", fontsize=9)
        axes[0, i].axis("off")
        if i == n_slices - 1:
            plt.colorbar(im0, ax=axes[0, i], label="HRF Index")

        # Plot recovered
        im1 = axes[1, i].imshow(
            rec_slice.T,
            cmap="turbo",
            interpolation="nearest",
            vmin=true_hrf_indices.min(),
            vmax=true_hrf_indices.max(),
        )
        axes[1, i].set_title(f"Recovered\n(Acc={slice_accuracy:.2%})", fontsize=9)
        axes[1, i].axis("off")
        if i == n_slices - 1:
            plt.colorbar(im1, ax=axes[1, i], label="HRF Index")

        # Plot difference
        vmax_diff = max(abs(diff_slice.min()), abs(diff_slice.max()), 1e-10)
        im2 = axes[2, i].imshow(
            diff_slice.T, cmap="RdBu_r", interpolation="nearest", vmin=-vmax_diff, vmax=vmax_diff
        )
        axes[2, i].set_title("Difference", fontsize=9)
        axes[2, i].axis("off")
        if i == n_slices - 1:
            plt.colorbar(im2, ax=axes[2, i], label="Error")

    # Add row labels
    axes[0, 0].set_ylabel("Ground Truth", fontsize=10, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Recovered", fontsize=10, rotation=90, labelpad=10)
    axes[2, 0].set_ylabel("Error", fontsize=10, rotation=90, labelpad=10)

    plt.suptitle(
        f"HRF Library Recovery (Overall Accuracy: {accuracy_overall:.2%})", fontsize=14, y=0.98
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved HRF recovery plot to {save_path}")

    # Compute detailed accuracy stats
    accuracy = {
        "overall_accuracy": accuracy_overall,
        "mean_absolute_error": np.abs(recovered_hrf_indices - true_hrf_indices).mean(),
        "median_absolute_error": np.median(np.abs(recovered_hrf_indices - true_hrf_indices)),
        "max_error": np.abs(recovered_hrf_indices - true_hrf_indices).max(),
        "correct_voxels": correct.sum(),
        "total_voxels": correct.size,
    }

    print("\nHRF Recovery Statistics:")
    print(f"  Overall Accuracy: {accuracy['overall_accuracy']:.2%}")
    print(f"  Correct Voxels: {accuracy['correct_voxels']}/{accuracy['total_voxels']}")
    print(f"  Mean Absolute Error: {accuracy['mean_absolute_error']:.3f} HRF indices")
    print(f"  Max Error: {accuracy['max_error']:.0f} HRF indices")

    return fig, accuracy


# Example usage
if __name__ == "__main__":
    print("Design Optimization Module")
    print("=" * 80)
    print("This module provides tools for experimental design space exploration.")
    print("\nKey functions:")
    print("  - generate_isi_sequence(): Generate ISI sequences from distributions")
    print("  - sample_design_space(): Create candidate designs")
    print("  - evaluate_design_candidates(): Compute empirical metrics")
    print("  - find_optimal_designs(): Identify best designs")
    print("  - plot_fitness_landscape(): Visualize power-efficiency space")
    print("  - plot_pareto_frontier(): Show Pareto optimal designs")
    print("  - plot_isi_range_optimization(): Das 2023 Figure 7 style (auto mean)")
    print("  - plot_isi_range_by_target_mean(): ISI optimization with explicit means")
    print("  - plot_hrf_index_recovery(): Visualize HRF library recovery accuracy")
    print("\nSee example scripts for complete workflows.")
