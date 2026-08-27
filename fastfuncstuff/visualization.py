"""
FastFuncSim Visualization Module

Comprehensive visualization tools for:
1. Single-case deep exploration (voxel-level analysis)
2. Batch simulation summaries (statistical power, efficiency)
3. Efficiency-power-entropy trade-offs (Liu & Frank 2004)
4. HRF recovery quality
5. Multi-axis parametric exploration (magnitudes x HRFs x noise)

Design Philosophy:
- Single case: Deep dive into how well the model performs
- Batch: Summary statistics (R², efficiency, estimability)
- Flexible: Support any number of conditions, orderings, magnitudes
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec


def plot_simulation_deep_dive(
    data: torch.Tensor | np.ndarray,
    design: torch.Tensor | np.ndarray,
    results,  # GLMResults object
    onsets: torch.Tensor | np.ndarray | None = None,
    betas_true: torch.Tensor | np.ndarray | None = None,
    hrf_true: torch.Tensor | np.ndarray | None = None,
    voxel_selection: str = "best",
    n_voxels: int = 4,
    tr: float = 1.0,
    figsize: tuple[int, int] = (20, 12),
    save_path: str | None = None,
):
    """
    Deep dive visualization of single simulation

    Shows:
    - Timecourses (observed, predicted, residuals)
    - Design matrix visualization
    - Beta estimates vs true (if provided)
    - R² and diagnostics
    - HRF recovery (if FIR mode)

    Parameters
    ----------
    data : array-like, shape (nx, ny, nz, nt) or (n_voxels, nt)
        Simulated fMRI data
    design : array-like, shape (nt, n_regressors)
        Design matrix
    results : GLMResults
        Results from fit_glm
    onsets : array-like, optional
        Onset matrix for visualization
    betas_true : array-like, optional
        True beta values for comparison
    hrf_true : array-like, optional
        True HRF for comparison
    voxel_selection : str, default='best'
        How to select voxels: 'best' (highest R²), 'random', 'worst', 'median'
    n_voxels : int, default=4
        Number of voxels to visualize
    tr : float, default=1.0
        Repetition time in seconds
    figsize : tuple, default=(20, 12)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    # Convert to numpy for plotting
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if torch.is_tensor(design):
        design = design.cpu().numpy()
    if torch.is_tensor(results.betas):
        betas = results.betas.cpu().numpy()
        r2 = results.r2.cpu().numpy()
    else:
        betas = results.betas
        r2 = results.r2

    # Reshape data if needed
    if data.ndim == 4:
        nx, ny, nz, nt = data.shape
        data = data.reshape(-1, nt)
    else:
        nt = data.shape[-1]

    n_voxels_total = data.shape[0]
    n_regressors = design.shape[1]

    # Select voxels to visualize
    if voxel_selection == "best":
        voxel_indices = np.argsort(r2)[-n_voxels:][::-1]
    elif voxel_selection == "worst":
        voxel_indices = np.argsort(r2)[:n_voxels]
    elif voxel_selection == "median":
        sorted_idx = np.argsort(r2)
        mid = len(sorted_idx) // 2
        start = max(0, mid - n_voxels // 2)
        voxel_indices = sorted_idx[start : start + n_voxels]
    elif voxel_selection == "random":
        voxel_indices = np.random.choice(n_voxels_total, n_voxels, replace=False)
    else:
        raise ValueError(f"Unknown voxel_selection: {voxel_selection}")

    # Get predicted timecourses
    predicted = design @ betas.T  # (nt, n_voxels)
    residuals = data.T - predicted  # (nt, n_voxels)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_voxels + 1, figure=fig, hspace=0.3, wspace=0.3)

    # Time axis
    time_axis = np.arange(nt) * tr

    # Plot individual voxels
    for i, vox_idx in enumerate(voxel_indices):
        # Row 1: Observed vs Predicted
        ax1 = fig.add_subplot(gs[0, i])
        ax1.plot(time_axis, data[vox_idx, :], "k-", alpha=0.5, linewidth=1, label="Observed")
        ax1.plot(time_axis, predicted[:, vox_idx], "r-", linewidth=2, label="Predicted")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Signal")
        ax1.set_title(f"Voxel {vox_idx}\nR²={r2[vox_idx]:.3f}")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Row 2: Residuals
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(time_axis, residuals[:, vox_idx], "b-", linewidth=1)
        ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Residual")
        ax2.set_title(f"Residuals (SD={residuals[:, vox_idx].std():.2f})")
        ax2.grid(True, alpha=0.3)

        # Row 3: Beta values
        ax3 = fig.add_subplot(gs[2, i])
        x_pos = np.arange(n_regressors)
        ax3.bar(x_pos, betas[vox_idx, :], alpha=0.7, color="steelblue", label="Estimated")

        if betas_true is not None:
            if torch.is_tensor(betas_true):
                betas_true_np = betas_true.cpu().numpy()
            else:
                betas_true_np = betas_true

            # Handle different shapes
            if betas_true_np.ndim == 1:
                # Same beta for all voxels
                ax3.scatter(
                    x_pos[: len(betas_true_np)],
                    betas_true_np,
                    color="red",
                    s=100,
                    zorder=10,
                    marker="x",
                    linewidths=3,
                    label="True",
                )
            elif betas_true_np.ndim == 2:
                # Different beta per voxel
                ax3.scatter(
                    x_pos,
                    betas_true_np[vox_idx, :],
                    color="red",
                    s=100,
                    zorder=10,
                    marker="x",
                    linewidths=3,
                    label="True",
                )

        ax3.set_xlabel("Regressor")
        ax3.set_ylabel("Beta")
        ax3.set_title("Parameter Estimates")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Right column: Summary statistics
    # Design matrix
    ax_design = fig.add_subplot(gs[0, -1])
    im = ax_design.imshow(
        design.T,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-design.std() * 3,
        vmax=design.std() * 3,
    )
    ax_design.set_xlabel("Time (TRs)")
    ax_design.set_ylabel("Regressor")
    ax_design.set_title("Design Matrix")
    plt.colorbar(im, ax=ax_design, fraction=0.046, pad=0.04)

    # R² histogram
    ax_r2_hist = fig.add_subplot(gs[1, -1])
    ax_r2_hist.hist(r2, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax_r2_hist.axvline(
        np.mean(r2), color="red", linestyle="--", linewidth=2, label=f"Mean={np.mean(r2):.3f}"
    )
    ax_r2_hist.axvline(
        np.median(r2),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median={np.median(r2):.3f}",
    )
    ax_r2_hist.set_xlabel("R²")
    ax_r2_hist.set_ylabel("Count")
    ax_r2_hist.set_title(f"R² Distribution (n={n_voxels_total})")
    ax_r2_hist.legend()
    ax_r2_hist.grid(True, alpha=0.3)

    # Summary statistics text
    ax_summary = fig.add_subplot(gs[2, -1])
    ax_summary.axis("off")

    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    summary_text += f"Data Shape: {data.shape}\n"
    summary_text += f"Design: {design.shape[0]} TRs × {design.shape[1]} regressors\n\n"
    summary_text += "R² Statistics:\n"
    summary_text += f"  Mean:   {np.mean(r2):.4f}\n"
    summary_text += f"  Median: {np.median(r2):.4f}\n"
    summary_text += f"  Std:    {np.std(r2):.4f}\n"
    summary_text += f"  Min:    {np.min(r2):.4f}\n"
    summary_text += f"  Max:    {np.max(r2):.4f}\n\n"

    if betas_true is not None:
        if betas_true_np.ndim == 1:
            beta_error = np.abs(betas[:, : len(betas_true_np)] - betas_true_np[np.newaxis, :])
        else:
            beta_error = np.abs(betas - betas_true_np)
        summary_text += "Beta Estimation Error:\n"
        summary_text += f"  MAE:  {np.mean(beta_error):.4f}\n"
        summary_text += f"  RMSE: {np.sqrt(np.mean(beta_error**2)):.4f}\n"

    ax_summary.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle("Single Simulation Deep Dive", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_batch_summary(
    results_list: list[dict],
    metrics: list[str] | None = None,
    group_by: str | None = None,
    figsize: tuple[int, int] = (16, 10),
    save_path: str | None = None,
):
    """
    Summary visualization for batch simulations

    Parameters
    ----------
    results_list : list of dict
        Each dict contains simulation results with keys like:
        - 'r2_mean', 'r2_median', 'r2_std'
        - 'beta_error_mae', 'beta_error_rmse'
        - 'hrf_correlation', 'hrf_rmse'
        - 'effect_size', 'noise_level', 'hrf_type' (grouping variables)
    metrics : list of str
        Which metrics to plot
    group_by : str, optional
        Variable to group by (e.g., 'effect_size', 'noise_level')
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    if metrics is None:
        metrics = ["r2", "beta_error", "hrf_recovery"]

    n_sims = len(results_list)

    # Extract data
    r2_means = [r.get("r2_mean", r.get("mean_r2", np.nan)) for r in results_list]
    _r2_medians = [r.get("r2_median", r.get("median_r2", np.nan)) for r in results_list]
    _r2_stds = [r.get("r2_std", 0) for r in results_list]

    # Create figure
    n_plots = len(metrics)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    plot_idx = 0

    # Plot R² statistics
    if "r2" in metrics:
        ax = axes[plot_idx]

        if group_by and group_by in results_list[0]:
            # Grouped visualization
            groups = sorted(set(r[group_by] for r in results_list))
            x_pos = np.arange(len(groups))

            means_by_group = []
            stds_by_group = []
            for g in groups:
                group_vals = [r["r2_mean"] for r in results_list if r[group_by] == g]
                means_by_group.append(np.mean(group_vals))
                stds_by_group.append(np.std(group_vals))

            ax.bar(
                x_pos,
                means_by_group,
                yerr=stds_by_group,
                alpha=0.7,
                capsize=5,
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{g:.2f}" if isinstance(g, float) else str(g) for g in groups])
            ax.set_xlabel(group_by.replace("_", " ").title())
        else:
            # Overall distribution
            ax.hist(r2_means, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
            ax.axvline(
                np.mean(r2_means),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={np.mean(r2_means):.3f}",
            )
            ax.legend()

        ax.set_ylabel("R² (Mean across voxels)")
        ax.set_title(f"R² Distribution\n{n_sims} simulations")
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot beta estimation error
    if "beta_error" in metrics:
        ax = axes[plot_idx]

        beta_maes = [r.get("beta_error_mae", np.nan) for r in results_list]
        beta_rmses = [r.get("beta_error_rmse", np.nan) for r in results_list]

        if not all(np.isnan(beta_maes)):
            if group_by and group_by in results_list[0]:
                groups = sorted(set(r[group_by] for r in results_list))
                x_pos = np.arange(len(groups))

                mae_by_group = []
                rmse_by_group = []
                for g in groups:
                    group_mae = [
                        r["beta_error_mae"]
                        for r in results_list
                        if r[group_by] == g and not np.isnan(r.get("beta_error_mae", np.nan))
                    ]
                    group_rmse = [
                        r["beta_error_rmse"]
                        for r in results_list
                        if r[group_by] == g and not np.isnan(r.get("beta_error_rmse", np.nan))
                    ]
                    mae_by_group.append(np.mean(group_mae) if group_mae else 0)
                    rmse_by_group.append(np.mean(group_rmse) if group_rmse else 0)

                width = 0.35
                ax.bar(
                    x_pos - width / 2,
                    mae_by_group,
                    width,
                    label="MAE",
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                )
                ax.bar(
                    x_pos + width / 2,
                    rmse_by_group,
                    width,
                    label="RMSE",
                    alpha=0.7,
                    color="coral",
                    edgecolor="black",
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{g:.2f}" if isinstance(g, float) else str(g) for g in groups])
                ax.set_xlabel(group_by.replace("_", " ").title())
            else:
                ax.scatter(beta_maes, beta_rmses, alpha=0.5, s=50)
                ax.plot([0, max(beta_maes)], [0, max(beta_maes)], "k--", alpha=0.3)
                ax.set_xlabel("MAE")
                ax.set_ylabel("RMSE")

            ax.set_title("Beta Estimation Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Plot HRF recovery quality
    if "hrf_recovery" in metrics:
        ax = axes[plot_idx]

        hrf_corrs = [r.get("hrf_correlation", np.nan) for r in results_list]
        _hrf_rmses = [r.get("hrf_rmse", np.nan) for r in results_list]

        if not all(np.isnan(hrf_corrs)):
            if group_by and group_by in results_list[0]:
                groups = sorted(set(r[group_by] for r in results_list))
                x_pos = np.arange(len(groups))

                corr_by_group = []
                for g in groups:
                    group_corr = [
                        r["hrf_correlation"]
                        for r in results_list
                        if r[group_by] == g and not np.isnan(r.get("hrf_correlation", np.nan))
                    ]
                    corr_by_group.append(np.mean(group_corr) if group_corr else 0)

                ax.bar(x_pos, corr_by_group, alpha=0.7, color="forestgreen", edgecolor="black")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{g:.2f}" if isinstance(g, float) else str(g) for g in groups])
                ax.set_xlabel(group_by.replace("_", " ").title())
                ax.set_ylim([0, 1])
            else:
                ax.hist(hrf_corrs, bins=30, alpha=0.7, color="forestgreen", edgecolor="black")
                ax.axvline(
                    np.mean(hrf_corrs),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean={np.mean(hrf_corrs):.3f}",
                )
                ax.legend()

            ax.set_ylabel("Correlation with True HRF")
            ax.set_title("HRF Recovery Quality")
            ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Statistical power curve (if effect_size is available)
    if "power" in metrics and "effect_size" in results_list[0]:
        ax = axes[plot_idx]

        effect_sizes = sorted(set(r["effect_size"] for r in results_list))
        r2_thresholds = [0.01, 0.05, 0.10, 0.20]

        for threshold in r2_thresholds:
            power = []
            for es in effect_sizes:
                es_sims = [r for r in results_list if r["effect_size"] == es]
                prop_above = np.mean([r["r2_mean"] > threshold for r in es_sims])
                power.append(prop_above)

            ax.plot(effect_sizes, power, marker="o", linewidth=2, label=f"R²>{threshold}")

        ax.axhline(0.8, color="k", linestyle="--", alpha=0.5, label="80% power")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Statistical Power")
        ax.set_title("Power Curves")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Batch Simulation Summary ({n_sims} simulations)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_parametric_exploration(
    results_grid: dict,
    x_var: str = "beta_ratio",
    y_var: str = "hrf_type",
    z_var: str = "noise_level",
    metric: str = "r2_mean",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
):
    """
    Visualize parametric exploration across three axes

    Typical use: Explore effect of beta combinations × HRF variations × noise levels

    Parameters
    ----------
    results_grid : dict
        Nested dict with structure: {z_val: {y_val: {x_val: result_dict}}}
        Where result_dict contains metrics like 'r2_mean', 'beta_error_mae', etc.
    x_var : str
        Variable name for x-axis (e.g., 'beta_ratio', 'effect_size')
    y_var : str
        Variable name for y-axis (e.g., 'hrf_type', 'hrf_index')
    z_var : str
        Variable name for different subplots (e.g., 'noise_level')
    metric : str
        Which metric to display (e.g., 'r2_mean', 'beta_error_mae')
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    # Get unique values for each dimension
    z_vals = sorted(results_grid.keys())
    n_z = len(z_vals)

    # Determine subplot layout
    n_cols = min(3, n_z)
    n_rows = int(np.ceil(n_z / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_z == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color map
    cmap = plt.cm.viridis

    for z_idx, z_val in enumerate(z_vals):
        ax = axes[z_idx]

        y_vals = sorted(results_grid[z_val].keys())
        x_vals = sorted(results_grid[z_val][y_vals[0]].keys())

        # Create 2D grid
        grid_data = np.zeros((len(y_vals), len(x_vals)))

        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                result = results_grid[z_val][y_val][x_val]
                grid_data[i, j] = result.get(metric, np.nan)

        # Plot heatmap
        im = ax.imshow(grid_data, aspect="auto", cmap=cmap, interpolation="nearest", origin="lower")

        # Set ticks
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels(
            [f"{x:.2f}" if isinstance(x, float) else str(x) for x in x_vals],
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels([f"{y:.2f}" if isinstance(y, float) else str(y) for y in y_vals])

        ax.set_xlabel(x_var.replace("_", " ").title())
        ax.set_ylabel(y_var.replace("_", " ").title())
        ax.set_title(f"{z_var.replace('_', ' ').title()} = {z_val}")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if not np.isnan(grid_data[i, j]):
                    _text = ax.text(
                        j,
                        i,
                        f"{grid_data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if grid_data[i, j] < np.nanmean(grid_data) else "black",
                        fontsize=8,
                    )

    # Hide unused subplots
    for i in range(n_z, len(axes)):
        axes[i].axis("off")

    metric_title = metric.replace("_", " ").title()
    plt.suptitle(f"Parametric Exploration: {metric_title}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_hrf_recovery(
    hrf_estimated: torch.Tensor | np.ndarray,
    hrf_true: torch.Tensor | np.ndarray,
    tr: float = 1.0,
    voxel_selection: str = "best",
    n_voxels: int = 6,
    figsize: tuple[int, int] = (15, 8),
    save_path: str | None = None,
):
    """
    Visualize HRF recovery quality from FIR estimation

    Parameters
    ----------
    hrf_estimated : array-like, shape (n_voxels, n_lags) or (n_lags,)
        Estimated HRF from FIR
    hrf_true : array-like, shape (n_lags,)
        True HRF used in simulation
    tr : float, default=1.0
        Repetition time
    voxel_selection : str, default='best'
        How to select voxels: 'best', 'random', 'worst', 'median'
    n_voxels : int, default=6
        Number of voxels to show
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    # Convert to numpy
    if torch.is_tensor(hrf_estimated):
        hrf_estimated = hrf_estimated.cpu().numpy()
    if torch.is_tensor(hrf_true):
        hrf_true = hrf_true.cpu().numpy()

    # Reshape if needed
    if hrf_estimated.ndim == 1:
        hrf_estimated = hrf_estimated[np.newaxis, :]

    n_voxels_total = hrf_estimated.shape[0]
    n_lags = min(hrf_estimated.shape[1], len(hrf_true))

    # Truncate to matching length
    hrf_estimated = hrf_estimated[:, :n_lags]
    hrf_true = hrf_true[:n_lags]

    # Normalize for comparison
    hrf_true_norm = hrf_true / np.abs(hrf_true).max()
    hrf_est_norm = hrf_estimated / np.abs(hrf_estimated).max(axis=1, keepdims=True)

    # Compute correlation with true HRF
    correlations = np.array(
        [np.corrcoef(hrf_true_norm, hrf_est_norm[i, :])[0, 1] for i in range(n_voxels_total)]
    )

    # Select voxels
    if voxel_selection == "best":
        voxel_indices = np.argsort(correlations)[-n_voxels:][::-1]
    elif voxel_selection == "worst":
        voxel_indices = np.argsort(correlations)[:n_voxels]
    elif voxel_selection == "median":
        sorted_idx = np.argsort(correlations)
        mid = len(sorted_idx) // 2
        start = max(0, mid - n_voxels // 2)
        voxel_indices = sorted_idx[start : start + n_voxels]
    elif voxel_selection == "random":
        voxel_indices = np.random.choice(n_voxels_total, n_voxels, replace=False)
    else:
        raise ValueError(f"Unknown voxel_selection: {voxel_selection}")

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_voxels + 1, figure=fig, hspace=0.3, wspace=0.3)

    time_axis = np.arange(n_lags) * tr

    # Plot individual voxels
    for i, vox_idx in enumerate(voxel_indices):
        ax = fig.add_subplot(gs[0, i])

        ax.plot(time_axis, hrf_true_norm, "k--", linewidth=2, label="True HRF")
        ax.plot(time_axis, hrf_est_norm[vox_idx, :], "r-", linewidth=2, label="Estimated")

        corr = correlations[vox_idx]
        rmse = np.sqrt(np.mean((hrf_true_norm - hrf_est_norm[vox_idx, :]) ** 2))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response (norm.)")
        ax.set_title(f"Voxel {vox_idx}\nr={corr:.3f}, RMSE={rmse:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary: Correlation histogram
    ax_hist = fig.add_subplot(gs[0, -1])
    ax_hist.hist(correlations, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax_hist.axvline(
        np.mean(correlations),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean={np.mean(correlations):.3f}",
    )
    ax_hist.axvline(
        np.median(correlations),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median={np.median(correlations):.3f}",
    )
    ax_hist.set_xlabel("Correlation with True HRF")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"HRF Recovery Quality\n(n={n_voxels_total})")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Bottom row: Mean estimated HRF ± std
    ax_mean = fig.add_subplot(gs[1, : n_voxels // 2 + 1])

    mean_hrf = np.mean(hrf_est_norm, axis=0)
    std_hrf = np.std(hrf_est_norm, axis=0)

    ax_mean.fill_between(
        time_axis,
        mean_hrf - std_hrf,
        mean_hrf + std_hrf,
        alpha=0.3,
        color="steelblue",
        label="±1 SD",
    )
    ax_mean.plot(time_axis, mean_hrf, "b-", linewidth=2, label="Mean Estimated")
    ax_mean.plot(time_axis, hrf_true_norm, "k--", linewidth=2, label="True HRF")

    ax_mean.set_xlabel("Time (s)")
    ax_mean.set_ylabel("Response (normalized)")
    ax_mean.set_title("Mean Estimated HRF Across All Voxels")
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.3)

    # Summary statistics text
    ax_stats = fig.add_subplot(gs[1, n_voxels // 2 + 1 :])
    ax_stats.axis("off")

    # Calculate peak timing error
    true_peak_idx = np.argmax(np.abs(hrf_true_norm))
    true_peak_time = true_peak_idx * tr

    peak_times = np.array(
        [np.argmax(np.abs(hrf_est_norm[i, :])) * tr for i in range(n_voxels_total)]
    )
    peak_errors = peak_times - true_peak_time

    stats_text = "HRF Recovery Statistics\n" + "=" * 35 + "\n\n"
    stats_text += "Correlation:\n"
    stats_text += f"  Mean:   {np.mean(correlations):.4f}\n"
    stats_text += f"  Median: {np.median(correlations):.4f}\n"
    stats_text += f"  Std:    {np.std(correlations):.4f}\n"
    stats_text += f"  Min:    {np.min(correlations):.4f}\n"
    stats_text += f"  Max:    {np.max(correlations):.4f}\n\n"

    rmses = np.array(
        [np.sqrt(np.mean((hrf_true_norm - hrf_est_norm[i, :]) ** 2)) for i in range(n_voxels_total)]
    )
    stats_text += "RMSE:\n"
    stats_text += f"  Mean:   {np.mean(rmses):.4f}\n"
    stats_text += f"  Median: {np.median(rmses):.4f}\n\n"

    stats_text += "Peak Timing Error (s):\n"
    stats_text += f"  True peak: {true_peak_time:.2f}s\n"
    stats_text += f"  Mean error: {np.mean(peak_errors):.2f}s\n"
    stats_text += f"  Std error:  {np.std(peak_errors):.2f}s\n"

    ax_stats.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle("HRF Recovery Analysis (FIR Estimation)", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_design_comparison(
    designs: dict[str, np.ndarray],
    labels: list[str] | None = None,
    tr: float = 1.0,
    figsize: tuple[int, int] = (16, 8),
    save_path: str | None = None,
):
    """
    Compare multiple design matrices visually

    Parameters
    ----------
    designs : dict
        dictionary mapping design names to design matrices (nt, n_regressors)
    labels : list of str, optional
        Condition labels for legend
    tr : float, default=1.0
        Repetition time
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    n_designs = len(designs)

    fig, axes = plt.subplots(2, n_designs, figsize=figsize)
    if n_designs == 1:
        axes = axes[:, np.newaxis]

    design_names = list(designs.keys())

    for i, design_name in enumerate(design_names):
        design = designs[design_name]
        if torch.is_tensor(design):
            design = design.cpu().numpy()

        nt, n_regressors = design.shape
        time_axis = np.arange(nt) * tr

        # Top: Timecourses
        ax_time = axes[0, i]
        for reg in range(n_regressors):
            label = labels[reg] if labels and reg < len(labels) else f"Reg {reg}"
            ax_time.plot(time_axis, design[:, reg], label=label, linewidth=1.5)

        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        ax_time.set_title(f"{design_name}\n(Timecourses)")
        ax_time.legend(fontsize=8, loc="upper right")
        ax_time.grid(True, alpha=0.3)

        # Bottom: Heatmap
        ax_heat = axes[1, i]
        im = ax_heat.imshow(
            design.T,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-np.std(design) * 3,
            vmax=np.std(design) * 3,
        )
        ax_heat.set_xlabel("Time (TRs)")
        ax_heat.set_ylabel("Regressor")
        ax_heat.set_title(f"{design_name}\n(Heatmap)")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    plt.suptitle("Design Matrix Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def create_interactive_summary_html(
    results_list: list[dict], output_path: str = "simulation_summary.html"
):
    """
    Create interactive HTML summary of batch simulations

    Useful for exploring large numbers of simulations without overwhelming plots

    Parameters
    ----------
    results_list : list of dict
        Simulation results
    output_path : str
        Where to save HTML file

    Returns
    -------
    str : Path to created HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastFuncSim Batch Simulation Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .summary-stats { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .metric { display: inline-block; margin: 10px 20px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
            .metric-label { font-size: 14px; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>FastFuncSim Batch Simulation Summary</h1>
    """

    # Overall statistics
    n_sims = len(results_list)
    r2_means = [r.get("r2_mean", r.get("mean_r2", 0)) for r in results_list]

    html_content += f"""
    <div class="summary-stats">
        <h2>Overall Statistics</h2>
        <div class="metric">
            <div class="metric-value">{n_sims}</div>
            <div class="metric-label">Total Simulations</div>
        </div>
        <div class="metric">
            <div class="metric-value">{np.mean(r2_means):.4f}</div>
            <div class="metric-label">Mean R²</div>
        </div>
        <div class="metric">
            <div class="metric-value">{np.median(r2_means):.4f}</div>
            <div class="metric-label">Median R²</div>
        </div>
        <div class="metric">
            <div class="metric-value">{np.std(r2_means):.4f}</div>
            <div class="metric-label">Std R²</div>
        </div>
    </div>
    """

    # Detailed results table
    html_content += """
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Simulation</th>
    """

    # Get all keys from first result
    if results_list:
        keys = sorted(results_list[0].keys())
        for key in keys:
            html_content += f"<th>{key.replace('_', ' ').title()}</th>"

    html_content += "</tr>"

    # Add rows
    for i, result in enumerate(results_list):
        html_content += f"<tr><td>{i + 1}</td>"
        for key in keys:
            value = result.get(key, "")
            if isinstance(value, float):
                html_content += f"<td>{value:.4f}</td>"
            else:
                html_content += f"<td>{value}</td>"
        html_content += "</tr>"

    html_content += """
    </table>
    </body>
    </html>
    """

    # Write file
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Created interactive HTML summary: {output_path}")
    return output_path


def plot_noise_pool_pca_scree(
    scree_ratio_per_run: list[np.ndarray | torch.Tensor | list[float]],
    variance_threshold: float | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot per-run PCA scree curves from noise-pool data."""
    ratios_np: list[np.ndarray] = []
    for ratios in scree_ratio_per_run:
        if torch.is_tensor(ratios):
            r = ratios.detach().cpu().numpy()
        else:
            r = np.asarray(ratios)
        if r.ndim == 0:
            continue
        ratios_np.append(r.astype(np.float64, copy=False))

    if len(ratios_np) == 0:
        raise ValueError("scree_ratio_per_run is empty")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ratios_np)))

    for run_idx, (ratios, color) in enumerate(zip(ratios_np, colors, strict=False)):
        x = np.arange(1, len(ratios) + 1)
        ax1.plot(x, ratios, color=color, linewidth=1.5, alpha=0.9, label=f"Run {run_idx + 1}")

        cum = np.cumsum(ratios)
        cum = np.clip(cum, 0.0, 1.0)
        ax2.plot(x, cum, color=color, linewidth=1.5, alpha=0.9, label=f"Run {run_idx + 1}")

    ax1.set_title("Noise-pool PCA scree (per-run explained variance ratio)", fontweight="bold")
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Cumulative explained variance")
    ax2.set_xlabel("Component index")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, alpha=0.3)

    if variance_threshold is not None:
        ax2.axhline(variance_threshold, color="gray", linestyle="--", alpha=0.8)

    if len(ratios_np) <= 10:
        ax1.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")

    return fig


# Multi-planar cut geometry for the denoising PC figures.
#
# Each plane is transposed so the anatomical superior axis runs up the page
# under origin="lower": rows are the second-named axis, columns the first.
_PLANE_SPECS = (
    ("Sagittal", 0, 2, 1),  # cut along x, rows = z, cols = y
    ("Coronal", 1, 2, 0),  # cut along y, rows = z, cols = x
    ("Axial", 2, 1, 0),  # cut along z, rows = y, cols = x
)


def _plane_slice(vol: np.ndarray, cut_ax: int, idx: int, row_ax: int) -> np.ndarray:
    """Take one 2-D cut with `row_ax` as the vertical (row) axis."""
    cut = np.take(vol, idx, axis=cut_ax)
    # np.take drops cut_ax, so the surviving axes keep their relative order.
    remaining = [a for a in range(3) if a != cut_ax]
    return cut if remaining[0] == row_ax else cut.T


def _cut_indices(mask_vol: np.ndarray | None, cut_ax: int, n_cut: int, extent: int) -> np.ndarray:
    """Evenly spaced cuts inside the mask's extent along `cut_ax`.

    The old code hardcoded a 5-voxel inset from each end, which inverts the
    range on a thin slab (a 12-slice slab has no interior left). Work in
    fractions of the mask extent instead so slabs and whole brains both land
    on interior cuts.
    """
    lo, hi = 0, extent - 1
    if mask_vol is not None:
        present = np.where(mask_vol.any(axis=tuple(a for a in range(3) if a != cut_ax)))[0]
        if present.size:
            lo, hi = int(present.min()), int(present.max())

    span = hi - lo
    fracs = (np.arange(n_cut) + 1.0) / (n_cut + 1.0)
    return np.clip(np.round(lo + fracs * span).astype(int), 0, extent - 1)


def _mask_bbox(mask_vol: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    """Tightest box containing the mask, padded by `pad` voxels per side.

    Without this the montage draws the whole acquisition box, and on a slab the
    empty surround is most of the picture.
    """
    box = []
    for ax in range(3):
        present = np.where(mask_vol.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        if present.size:
            lo = max(0, int(present.min()) - pad)
            hi = min(mask_vol.shape[ax], int(present.max()) + 1 + pad)
        else:
            lo, hi = 0, mask_vol.shape[ax]
        box.append(slice(lo, hi))
    return box[0], box[1], box[2]


def _montage_2d(cuts: list[np.ndarray], pad: int = 2) -> np.ndarray:
    """Lay cuts of one plane side by side with a NaN gutter between them."""
    if len(cuts) == 1:
        return cuts[0]
    gutter = np.full((cuts[0].shape[0], pad), np.nan)
    pieces: list[np.ndarray] = []
    for i, cut in enumerate(cuts):
        if i:
            pieces.append(gutter)
        pieces.append(cut)
    return np.hstack(pieces)


def plot_denoising_pcs(
    noise_pcs_per_run: list[torch.Tensor | np.ndarray],
    run_starts: list[int],
    component_variance_ratio_per_run: list[torch.Tensor | np.ndarray] | None = None,
    pc_weights_per_run: list[np.ndarray] | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    voxel_mask: np.ndarray | None = None,
    noise_pool_mask: np.ndarray | None = None,
    n_pcs_to_show: int = 5,
    n_slices: int = 3,
    slice_axis: str = "x",
    tr: float = 2.0,
    optimal_n_pcs: int | None = None,
    output_prefix: str | None = None,
    voxel_sizes: tuple[float, float, float] | None = None,
    return_figs: bool = True,
) -> list[plt.Figure]:
    """
    Create tedana-style visualization of noise PCs for denoising diagnostics.

    For each PC (up to n_pcs_to_show), creates a figure showing:
    - Top: Full-width PC timecourse (run-concatenated)
    - Bottom: Spatial slices organized by run (one column per run)

    Parameters
    ----------
    noise_pcs_per_run : list of Tensor or ndarray
        PC timecourses per run. Each has shape (n_timepoints_run, n_components).
    run_starts : list of int
        Starting timepoint for each run.
    component_variance_ratio_per_run : list, optional
        Per-run variance-share vectors for components, where each run entry has
        shape (n_components,). Used to annotate each component with a small
        variance-share text label (e.g., "12.34%").
    pc_weights_per_run : list of ndarray, optional
        Spatial weights per run per PC. Each has shape (n_noise_voxels, n_components).
        If provided, shows brain slices of weights.
    volume_shape : tuple of int, optional
        3D volume shape (nx, ny, nz) for reshaping weights to volume.
    voxel_mask : ndarray, optional
        Boolean mask of shape (nx*ny*nz,) for brain voxels.
    noise_pool_mask : ndarray, optional
        Boolean mask of shape (n_brain_voxels,) for noise pool voxels.
        Required when pc_weights_per_run is provided.
    n_pcs_to_show : int, default=5
        Number of PCs to visualize.
    n_slices : int, default=3
        Number of cuts per plane, per run. All three planes are always drawn.
    slice_axis : str, default='x'
        Which plane is drawn first ('x' = sagittal, 'y' = coronal, 'z' = axial).
        The other two follow; no axis is ever dropped, which is what made thin
        slabs unreadable when a single axis was chosen.
    tr : float, default=2.0
        Repetition time in seconds (for x-axis time labels).
    optimal_n_pcs : int, optional
        Optimal number of PCs from cross-validation (for annotation).
    output_prefix : str, optional
        If provided, save figures to {output_prefix}_PC{n}.png.
    return_figs : bool, default=True
        Whether to return figure handles. Set to False when saving many plots
        to avoid retaining open figures in memory.
    voxel_sizes : tuple of float, optional
        Voxel sizes in mm (sx, sy, sz). If provided, preserves physical aspect ratio.

    Returns
    -------
    figs : list of Figure
        list of matplotlib figures, one per PC.

    Examples
    --------
    >>> figs = plot_denoising_pcs(
    ...     noise_pcs_per_run=results.noise_pcs_per_run,
    ...     run_starts=[0, 147, 300, 453],
    ...     tr=2.5,
    ...     slice_axis='x',
    ...     optimal_n_pcs=results.optimal_n_components,
    ...     output_prefix="subject01_denoising"
    ... )
    """
    # Convert tensors to numpy
    pcs_np = []
    for pcs in noise_pcs_per_run:
        if torch.is_tensor(pcs):
            pcs_np.append(pcs.cpu().numpy())
        else:
            pcs_np.append(pcs)

    n_runs = len(pcs_np)

    var_ratio_np = None
    if component_variance_ratio_per_run is not None:
        var_ratio_np = []
        for ratios in component_variance_ratio_per_run:
            if torch.is_tensor(ratios):
                var_ratio_np.append(ratios.detach().cpu().numpy())
            else:
                var_ratio_np.append(np.asarray(ratios))

    # Get max components across all runs
    max_pcs_available = min(pc.shape[1] for pc in pcs_np)
    n_pcs_to_show = min(n_pcs_to_show, max_pcs_available)

    # All three planes are always drawn; slice_axis only picks which leads.
    lead_ax = {"x": 0, "y": 1, "z": 2}.get(slice_axis.lower(), 0)
    plane_specs = sorted(_PLANE_SPECS, key=lambda spec: spec[1] != lead_ax)

    # Mask geometry is the same for every PC, so resolve it once. Cropping to the
    # mask box is what keeps the brain, not the surround, at figure scale.
    mask_vol = None
    crop = (slice(None), slice(None), slice(None))
    disp_shape = tuple(volume_shape) if volume_shape is not None else None

    if volume_shape is not None and voxel_mask is not None:
        # The support of the weights, built by the same two-level scatter the
        # weights themselves go through. Where the PCs were never estimated
        # (brain voxels outside the noise pool) is background, not zero loading.
        support = np.zeros(int(np.prod(volume_shape)), dtype=bool)
        if noise_pool_mask is not None:
            brain = np.zeros(int(voxel_mask.sum()), dtype=bool)
            brain[noise_pool_mask] = True
            support[voxel_mask] = brain
        else:
            support[voxel_mask] = True
        mask_vol = support.reshape(volume_shape)
        crop = _mask_bbox(mask_vol)
        mask_vol = mask_vol[crop]
        disp_shape = mask_vol.shape

    figs = []

    for pc_idx in range(n_pcs_to_show):
        # Determine if this PC is included in optimal model
        is_in_optimal = optimal_n_pcs is not None and pc_idx < optimal_n_pcs
        pc_label = f"PC {pc_idx + 1}"
        if optimal_n_pcs is not None:
            status = "✓ INCLUDED" if is_in_optimal else "✗ NOT INCLUDED"
            pc_label += f" ({status} in optimal {optimal_n_pcs}-PC model)"

        # Create figure layout:
        # Row 0: full-width timecourse, spanning every run column
        # Rows 1-3: one plane (sagittal / coronal / axial) per row; column r
        #           holds run r's cuts, so they sit under that run's timecourse
        has_weights = pc_weights_per_run is not None and volume_shape is not None

        run_lengths = [pc.shape[0] for pc in pcs_np]
        total_tps = sum(run_lengths)
        _time_axis = np.arange(total_tps) * tr

        fig_w = 32.0
        tc_h = 4.0

        if has_weights:
            sx, sy, sz = voxel_sizes if voxel_sizes is not None else (1.0, 1.0, 1.0)
            mm = (float(sx), float(sy), float(sz))

            # Each plane row is sized from the physical shape of its montage, so a
            # thin slab gets a short-and-wide sagittal row instead of being
            # squashed into an equal-height one. Size against the narrowest run
            # column so no row overflows the figure.
            min_col_w = fig_w * (min(run_lengths) / total_tps)
            plane_row_h = []
            for _name, _cut_ax, row_ax, col_ax in plane_specs:
                h_mm = disp_shape[row_ax] * mm[row_ax]
                w_mm = n_slices * disp_shape[col_ax] * mm[col_ax]
                plane_row_h.append(max(0.6, min_col_w * h_mm / w_mm))

            # Two gridspecs rather than one four-row grid: the timecourse needs
            # room for its axis labels while the plane rows want to sit tight
            # against each other, and a single hspace cannot do both.
            pane_h = sum(plane_row_h)
            fig_h = tc_h + pane_h + 1.4
            fig = plt.figure(figsize=(fig_w, fig_h))

            split = (pane_h + 0.5) / fig_h
            gs_tc = fig.add_gridspec(1, 1, top=1.0 - 0.7 / fig_h, bottom=split + 0.35 / fig_h)
            gs = fig.add_gridspec(
                3,
                n_runs,
                top=split,
                bottom=0.3 / fig_h,
                height_ratios=plane_row_h,
                width_ratios=run_lengths,
                hspace=0.14,
                wspace=0.02,
            )
            ax_tc = fig.add_subplot(gs_tc[0, 0])
        else:
            fig, ax_tc = plt.subplots(1, 1, figsize=(fig_w, tc_h))

        # Plot each run with different color
        colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
        current_tp = 0

        for run_idx, (pcs, color) in enumerate(zip(pcs_np, colors, strict=False)):
            run_length = pcs.shape[0]
            run_time = np.arange(run_length) * tr + current_tp * tr
            ax_tc.plot(
                run_time,
                pcs[:, pc_idx],
                color=color,
                linewidth=1.0,
                label=f"Run {run_idx + 1}",
                alpha=0.8,
            )
            current_tp += run_length

        ax_tc.set_xlabel("Time (s)", fontsize=11)
        ax_tc.set_ylabel("PC Amplitude (a.u.)", fontsize=11)
        ax_tc.set_title("Run-concatenated timecourse", fontsize=11)
        ax_tc.set_xlim(0, total_tps * tr)
        ax_tc.grid(True, alpha=0.3)

        # Add run boundaries
        for _run_idx, start in enumerate(run_starts[1:], 1):
            run_time = start * tr
            ax_tc.axvline(run_time, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)

        # Small per-run variance-share annotation (if provided)
        if var_ratio_np is not None and n_runs > 0:
            y_min, y_max = ax_tc.get_ylim()
            y_pos = y_min + 0.90 * (y_max - y_min)

            fallback_starts = np.cumsum([0] + [pc.shape[0] for pc in pcs_np[:-1]])

            for run_idx in range(n_runs):
                if run_idx >= len(var_ratio_np):
                    continue
                v = var_ratio_np[run_idx]
                if v is None or v.ndim == 0 or pc_idx >= len(v):
                    continue

                pct = float(v[pc_idx]) * 100.0
                run_start = (
                    run_starts[run_idx]
                    if run_idx < len(run_starts)
                    else int(fallback_starts[run_idx])
                )
                run_len = pcs_np[run_idx].shape[0]
                x_center = (run_start + run_len / 2.0) * tr

                ax_tc.text(
                    x_center,
                    y_pos,
                    f"{pct:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="dimgray",
                    bbox=dict(
                        boxstyle="round,pad=0.10", facecolor="white", alpha=0.40, edgecolor="none"
                    ),
                    zorder=4,
                )

        # Color the background differently if included vs not
        if is_in_optimal:
            ax_tc.set_facecolor("#e6ffe6")  # Light green
        else:
            ax_tc.set_facecolor("#ffe6e6")  # Light red

        # --- Bottom panels: multi-planar cuts, one column per run ---
        # Column r sits directly beneath run r's segment of the timecourse
        # (GridSpec width_ratios are the run lengths), so a spatial pattern can
        # be read against the timecourse wobble that produced it.
        if has_weights:
            # One scale for every run and plane of this PC, so brightness
            # differences between columns mean something.
            all_w = np.concatenate(
                [np.abs(w[:, pc_idx]) for w in pc_weights_per_run if pc_idx < w.shape[1] and w.size]
            )
            vmax = float(np.percentile(all_w, 98)) if all_w.size else 1.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0

            cmap = plt.get_cmap("RdBu_r").copy()
            # Out-of-mask voxels and the gutters between cuts are NaN. They have
            # to be a colour the map cannot produce: zero weight is white in
            # RdBu_r, so leaving background at zero made "no data" and "no
            # loading" identical.
            cmap.set_bad(color="black")

            for run_idx, weights in enumerate(pc_weights_per_run):
                if run_idx >= n_runs or pc_idx >= weights.shape[1]:
                    continue

                # Reshape weights to volume (two-level masking)
                run_weights = weights[:, pc_idx]
                vol = np.zeros(int(np.prod(volume_shape)))

                # TODO - fit PCs to whole brain mask - so we can see how they fit in all areas and
                # plot all voxels, not just noise pool. We would also want to save those nii (4d, per run, of pcs)
                if voxel_mask is not None and noise_pool_mask is not None:
                    # Create intermediate brain volume
                    brain_vol = np.zeros(voxel_mask.sum())
                    brain_vol[noise_pool_mask] = run_weights
                    vol[voxel_mask] = brain_vol
                elif voxel_mask is not None:
                    vol[voxel_mask] = run_weights
                else:
                    vol = run_weights
                vol = vol.reshape(volume_shape)[crop]
                if mask_vol is not None:
                    vol = np.where(mask_vol, vol, np.nan)

                for plane_idx, (name, cut_ax, row_ax, col_ax) in enumerate(plane_specs):
                    idxs = _cut_indices(mask_vol, cut_ax, n_slices, disp_shape[cut_ax])
                    montage = _montage_2d([_plane_slice(vol, cut_ax, int(i), row_ax) for i in idxs])

                    ax = fig.add_subplot(gs[plane_idx, run_idx])
                    ax.axis("off")
                    ax.imshow(
                        montage,
                        cmap=cmap,
                        vmin=-vmax,
                        vmax=vmax,
                        # Data aspect is mm-per-row over mm-per-column; the old
                        # code folded in the voxel *counts* too and stretched
                        # every non-cubic volume.
                        aspect=mm[row_ax] / mm[col_ax],
                        origin="lower",
                        interpolation="nearest",
                    )

                    if plane_idx == 0:
                        ax.set_title(f"Run {run_idx + 1}", fontsize=11, fontweight="bold")
                    if run_idx == 0:
                        ax.text(
                            -0.01,
                            0.5,
                            name,
                            transform=ax.transAxes,
                            rotation=90,
                            ha="right",
                            va="center",
                            fontsize=10,
                            color="dimgray",
                        )

        plt.suptitle(pc_label, fontsize=14, fontweight="bold", y=0.98)

        if output_prefix:
            fig_path = f"{output_prefix}_PC{pc_idx + 1:02d}.png"
            # Ensure parent directory exists
            Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")

        if return_figs:
            figs.append(fig)
        else:
            plt.close(fig)

    return figs


def plot_denoising_summary(
    xval_r2_by_n_components: np.ndarray,
    xval_r2_per_fold: np.ndarray,
    optimal_n_components: int,
    initial_r2_distribution: np.ndarray | None = None,
    r2_threshold: float = 0.05,
    n_noise_voxels: int | None = None,
    n_criteria_voxels: int | None = None,
    xval_r2_all_voxels: np.ndarray | None = None,
    min_gain: float | None = None,
    n_cv_folds: int | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create comprehensive summary plot for denoising cross-validation.

    Shows:
    - CV R² vs number of PCs (mean and per-fold)
    - Optimal PC count annotation
    - Initial R² distribution with threshold
    - Summary statistics

    Parameters
    ----------
    xval_r2_by_n_components : ndarray, shape (max_components + 1,)
        Mean cross-validated R² for each number of components.
    xval_r2_per_fold : ndarray, shape (n_folds, max_components + 1)
        R² for each CV fold and number of components.
    optimal_n_components : int
        Optimal number of components from cross-validation.
    initial_r2_distribution : ndarray, optional
        Initial R² values for all voxels (for histogram).
    r2_threshold : float, default=0.05
        R² threshold used for noise pool selection.
    n_noise_voxels : int, optional
        Number of voxels in noise pool.
    n_criteria_voxels : int, optional
        Number of voxels in criteria mask.
    xval_r2_all_voxels : ndarray, optional
        The same curve over every voxel rather than the criteria subset. Drawn
        on a twin axis: the selection curve rising while this one falls is the
        signature of selection bias, not of denoising.
    n_cv_folds : int, optional
        How many CV folds actually ran. The scoring convention concatenates
        every held-out run's predictions and takes ONE R2 per voxel, so
        `xval_r2_per_fold` has a single row no matter how many runs there
        were — deriving the fold count from its shape reported "1 fold" on a
        six-run dataset.
    min_gain : float, optional
        Gain floor the selection had to clear. Drawn as a band above baseline so
        a flat curve is visibly flat instead of auto-scaled into a landscape.
    output_path : str, optional
        If provided, save figure to this path.
    figsize : tuple, default=(14, 10)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    max_components = len(xval_r2_by_n_components) - 1
    n_rows = xval_r2_per_fold.shape[0]
    # One row means the folds were concatenated before scoring, not that one
    # fold ran. Only trust the row count as a fold count when there are several.
    n_folds = n_cv_folds if n_cv_folds is not None else n_rows
    scored_per_fold = n_rows > 1

    # --- Panel 1: CV R² by number of PCs ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot per-fold lines (thin, transparent) with a single legend entry
    if scored_per_fold:
        for fold_idx in range(n_rows):
            label = f"Per-fold R² (n={n_rows})" if fold_idx == 0 else "_nolegend_"
            ax1.plot(
                range(max_components + 1),
                xval_r2_per_fold[fold_idx, :],
                color="gray",
                alpha=0.4,
                linewidth=1,
                label=label,
            )

    # Plot median/mean (thick)
    ax1.plot(
        range(max_components + 1),
        xval_r2_by_n_components,
        "b-o",
        linewidth=2,
        markersize=6,
        label="Median R² (criteria voxels)",
    )

    # Mark optimal
    ax1.axvline(
        optimal_n_components,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_n_components} PCs",
    )
    ax1.scatter(
        [optimal_n_components],
        [xval_r2_by_n_components[optimal_n_components]],
        color="red",
        s=100,
        zorder=5,
        marker="*",
    )

    # The gain floor, drawn as a band. Without it an auto-scaled y-axis makes a
    # 2e-4 wobble on a 0.11 baseline look like a real selection curve.
    if min_gain is not None:
        base = float(xval_r2_by_n_components[0])
        ax1.axhspan(
            base,
            base + float(min_gain),
            color="gray",
            alpha=0.15,
            label=f"Below min gain ({min_gain:.2g})",
        )
        # Keep the band in view even when the curve never reaches it.
        lo, hi = ax1.get_ylim()
        ax1.set_ylim(min(lo, base - 0.2 * min_gain), max(hi, base + 1.2 * min_gain))

    if xval_r2_all_voxels is not None:
        ax1b = ax1.twinx()
        ax1b.plot(
            range(len(xval_r2_all_voxels)),
            np.asarray(xval_r2_all_voxels),
            color="darkorange",
            linewidth=1.5,
            linestyle=":",
            label="All voxels (right axis)",
        )
        # No y-label: it collides with the neighbouring panel's. The legend says it.
        ax1b.tick_params(axis="y", labelcolor="darkorange", labelsize=8)
        ax1b.legend(loc="upper right", fontsize=8)

    ax1.set_xlabel("Number of Noise PCs")
    ax1.set_ylabel("Cross-Validated R²")
    fold_note = (
        f"{n_folds} folds, predictions concatenated"
        if not scored_per_fold
        else f"{n_folds} folds, scored separately"
    )
    ax1.set_title(f"Denoising Performance (LORO CV; {fold_note})")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2 ---
    ax2 = fig.add_subplot(gs[0, 1])

    if scored_per_fold:
        # Heatmap of R² per fold/PC
        im = ax2.imshow(xval_r2_per_fold, aspect="auto", cmap="viridis")
        ax2.set_xlabel("Number of Noise PCs")
        ax2.set_ylabel("CV Fold (Run)")
        ax2.set_title("R² per Fold × PC Count")
        ax2.set_yticks(range(n_rows))
        ax2.set_yticklabels([f"Run {i + 1}" for i in range(n_rows)])

        # Mark optimal column
        ax2.axvline(optimal_n_components - 0.5, color="red", linewidth=2)
        ax2.axvline(optimal_n_components + 0.5, color="red", linewidth=2)

        plt.colorbar(im, ax=ax2, label="R²", shrink=0.8)
    else:
        # A single-row heatmap is just the panel-1 curve rendered as colour. Show
        # the decision instead: gain over baseline against the floor it must clear.
        gain = np.asarray(xval_r2_by_n_components) - float(xval_r2_by_n_components[0])
        ax2.plot(range(max_components + 1), gain, "b-o", linewidth=2, markersize=5)
        ax2.axhline(0, color="black", linewidth=1)

        if min_gain is not None:
            ax2.axhspan(
                -abs(float(min_gain)) * 0.25,
                float(min_gain),
                color="gray",
                alpha=0.18,
                label=f"Below min gain ({min_gain:.2g})",
            )

        if xval_r2_all_voxels is not None:
            all_v = np.asarray(xval_r2_all_voxels)
            ax2.plot(
                range(len(all_v)),
                all_v - float(all_v[0]),
                color="darkorange",
                linestyle=":",
                linewidth=1.5,
                label="All voxels",
            )

        ax2.axvline(
            optimal_n_components,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Selected: {optimal_n_components} PCs",
        )
        ax2.set_xlabel("Number of Noise PCs")
        ax2.set_ylabel("Δ R² vs 0 PCs")
        ax2.set_title("Gain over baseline (the selection criterion)")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

    # --- Panel 3: Initial R² distribution ---
    ax3 = fig.add_subplot(gs[1, 0])

    if initial_r2_distribution is not None:
        r2_vals = np.asarray(initial_r2_distribution)
        finite_mask = np.isfinite(r2_vals)
        r2_vals = r2_vals[finite_mask]
        ax3.hist(r2_vals, bins=50, alpha=0.7, edgecolor="black")
        ax3.axvline(
            r2_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {r2_threshold:.2f}",
        )
        ax3.set_xlabel("Initial R²")
        ax3.set_ylabel("Number of Voxels")
        ax3.set_yscale("log")
        ax3.set_title("R² Distribution (Noise Pool Selection)")
        ax3.legend()

        # Shade noise pool region
        ylim = ax3.get_ylim()
        ax3.fill_between(
            [0, r2_threshold], [ylim[1]] * 2, alpha=0.2, color="blue", label="Noise Pool"
        )
        ax3.fill_between(
            [r2_threshold, 1], [ylim[1]] * 2, alpha=0.2, color="green", label="Criteria"
        )
    else:
        ax3.text(0.5, 0.5, "No R² distribution data", ha="center", va="center")
        ax3.set_title("R² Distribution")

    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Summary text ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    baseline_r2 = xval_r2_by_n_components[0]
    optimal_r2 = xval_r2_by_n_components[optimal_n_components]
    improvement = optimal_r2 - baseline_r2
    improvement_pct = (improvement / baseline_r2) * 100 if baseline_r2 > 0 else 0

    summary_text = f"""
╔══════════════════════════════════════════════╗
║         DENOISING SUMMARY                    ║
╠══════════════════════════════════════════════╣
║  Cross-Validation:                           ║
║    • Folds: {n_folds} (leave-one-run-out)              ║
║    • Max PCs tested: {max_components}                       ║
║                                              ║
║  Voxel Selection:                            ║
║    • Noise pool: {n_noise_voxels:,} voxels              ║
║    • Criteria: {n_criteria_voxels:,} voxels                ║
║    • Threshold: R² < {r2_threshold:.2f}                    ║
║                                              ║
║  Performance:                                ║
║    • Baseline R² (0 PCs): {baseline_r2:.4f}            ║
║    • Optimal R² ({optimal_n_components} PCs): {optimal_r2:.4f}             ║
║    • Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)         ║
╚══════════════════════════════════════════════╝
"""

    ax4.text(
        0.05,
        0.95,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle("Cross-Validated Denoising Results", fontsize=14, fontweight="bold")

    if output_path:
        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
