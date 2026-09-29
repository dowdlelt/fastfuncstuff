"""
Event-sequence and ISI generators for designing experiments.

The primitives an experiment description is realized from -- the order of
trial types (generate_event_sequence) and the jittered intervals between them
(generate_isi_sequence) -- plus the TR-grid onset matrix. Design evaluation
lives in simulation/: experiment.py describes and realizes designs,
power.py simulates their power, metrics.py holds Liu & Frank (2004)
efficiency/power/entropy. The candidate-sampling optimizer that used to live
here was superseded by those and removed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import expon, poisson, truncexpon


@dataclass
class ISIConstraints:
    """Constraints for ISI distribution generation"""

    min_isi: float  # Minimum ISI in seconds
    max_isi: float  # Maximum ISI in seconds
    mean_isi: float  # Target mean ISI in seconds
    tr: float = 1.0  # Repetition time in seconds


def generate_event_sequence(
    n_trials_per_condition: int | list[int],
    n_conditions: int,
    ordering: Literal["random", "alternating", "blocked", "permuted_block"] = "random",
    block_size: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
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
        seed: Random seed (reseeds numpy's global RNG; prefer ``rng``)
        rng: Generator to draw from without touching global state, so several
            design realizations can be made independently and reproducibly

    Returns:
        event_sequence: Array of condition indices [0, 1, 0, 2, 1, ...]
    """
    if rng is None:
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(np.random.randint(0, 2**31))

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
        rng.shuffle(event_sequence)
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
            rng.shuffle(mini_block)
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
    rng: np.random.Generator | None = None,
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
            - 'poisson_target_mean': Poisson with a tighter (0.1%) mean-matching tolerance
            - 'uniform': Uniform random intervals
            - 'fixed': Fixed ISI (constant)
        seed: Random seed for reproducibility (reseeds numpy's global RNG)
        rng: Generator to draw from instead, leaving global state alone

    Returns:
        isis: Array of ISIs in seconds (length = n_events - 1)

    Strategy:
        1. Generate candidate ISIs from distribution
        2. Clip to [min_isi, max_isi] (or use truncated distribution)
        3. Iteratively adjust to match target mean

    Notes:
        - 'truncated_exponential': Uses scipy.stats.truncexpon for proper truncation
        - 'poisson_target_mean': Mean matched to 0.1% (the others to 1%)
        - 'uniform' and 'fixed' are not mean-adjusted; 'uniform' has mean
          (min_isi + max_isi) / 2 regardless of mean_isi.
    """
    if rng is None:
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(np.random.randint(0, 2**31))

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
        isis = expon.rvs(scale=scale, size=n_isis * 2, random_state=rng)  # Oversample for clipping

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
        isis_standardized = truncexpon.rvs(b=b_param, scale=1.0, size=n_isis * 2, random_state=rng)
        # Transform to [min_isi, max_isi]
        isis = min_isi + isis_standardized * scale_guess
        isis = np.clip(isis, min_isi, max_isi)  # Ensure bounds

    elif distribution == "poisson":
        # Poisson ISIs (discrete count → continuous time)
        lam = target_mean / isi_constraints.tr
        counts = poisson.rvs(mu=lam, size=n_isis * 2, random_state=rng)
        isis = counts * isi_constraints.tr
        isis = isis[isis > 0]  # Remove zero ISIs

    elif distribution == "poisson_target_mean":
        # Poisson with aggressive mean matching
        # Strategy: Generate Poisson samples, then use tighter tolerance in adjustment
        lam = target_mean / isi_constraints.tr
        counts = poisson.rvs(mu=lam, size=n_isis * 3, random_state=rng)  # Extra oversampling
        isis = counts * isi_constraints.tr
        isis = isis[isis > 0]  # Remove zero ISIs

        # Mean matching is left to the adjustment loop below. Pre-selecting the
        # draws closest to the target (as this once did) both collapsed the
        # jitter -- SD 0.74 s where the plain exponential gave 3.6 s -- and
        # returned them sorted by that distance, so the ISI sequence ran from
        # near-constant to extreme across the run.
        isis = np.clip(isis, min_isi, max_isi)

    elif distribution == "uniform":
        # Uniform distribution
        isis = rng.uniform(min_isi, max_isi, size=n_isis)

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
        warnings.warn(
            f"Only generated {len(isis)} valid ISIs, need {n_isis}. Padding with mean.",
            stacklevel=2,
        )
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

        # Rescale the excess over min_isi. Exponential and Poisson ISIs are a
        # shift-scale family above the floor, so this keeps their shape; only
        # the draws clipped at max_isi stop moving, which the next pass absorbs.
        # The additive nudge this replaced stalled once clipping pinned draws at
        # min_isi -- a Poisson(5) request with min 2 settled 1.4% high.
        for _i in range(max_iters):
            current_mean = isis.mean()
            if abs(target_mean - current_mean) < tolerance * target_mean:
                break
            excess = isis - min_isi
            # Shrinking moves every draw; stretching cannot move one at the cap.
            free = isis < max_isi if current_mean < target_mean else np.ones_like(isis, bool)
            if not free.any() or excess[free].sum() <= 0:
                break
            needed = target_mean * n_isis - isis[~free].sum() - min_isi * free.sum()
            isis = np.where(free, min_isi + excess * (needed / excess[free].sum()), isis)
            isis = np.clip(isis, min_isi, max_isi)

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

    # Convert to TRs and mark onsets. Events that do not fit, or that round
    # onto a TR already holding an event, are lost from the design -- say so,
    # since the trial counts the caller asked for are then not the ones scored.
    n_dropped = 0
    n_merged = 0
    for event_idx, onset_time in enumerate(onset_times):
        onset_tr = int(np.round(onset_time / tr))
        if onset_time >= duration or onset_tr >= n_timepoints:
            n_dropped += 1
            continue
        if onsets[onset_tr].any():
            n_merged += 1
        condition = event_sequence[event_idx]
        onsets[onset_tr, condition] = 1.0

    if n_dropped or n_merged:
        warnings.warn(
            f"create_onset_matrix: {n_dropped} of {len(onset_times)} events fall past the "
            f"{duration:g} s scan and {n_merged} share a TR with an earlier event "
            f"(ISIs below TR={tr:g} s round together on this grid)",
            stacklevel=2,
        )
    return onsets


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
