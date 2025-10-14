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

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Union, Tuple
import warnings


def plot_simulation_deep_dive(
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    results,  # GLMResults object
    onsets: Optional[Union[torch.Tensor, np.ndarray]] = None,
    betas_true: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hrf_true: Optional[Union[torch.Tensor, np.ndarray]] = None,
    voxel_selection: str = 'best',
    n_voxels: int = 4,
    tr: float = 1.0,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
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
    if voxel_selection == 'best':
        voxel_indices = np.argsort(r2)[-n_voxels:][::-1]
    elif voxel_selection == 'worst':
        voxel_indices = np.argsort(r2)[:n_voxels]
    elif voxel_selection == 'median':
        sorted_idx = np.argsort(r2)
        mid = len(sorted_idx) // 2
        start = max(0, mid - n_voxels // 2)
        voxel_indices = sorted_idx[start:start + n_voxels]
    elif voxel_selection == 'random':
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
        ax1.plot(time_axis, data[vox_idx, :], 'k-', alpha=0.5, linewidth=1, label='Observed')
        ax1.plot(time_axis, predicted[:, vox_idx], 'r-', linewidth=2, label='Predicted')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')
        ax1.set_title(f'Voxel {vox_idx}\nR²={r2[vox_idx]:.3f}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Row 2: Residuals
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(time_axis, residuals[:, vox_idx], 'b-', linewidth=1)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Residual')
        ax2.set_title(f'Residuals (SD={residuals[:, vox_idx].std():.2f})')
        ax2.grid(True, alpha=0.3)

        # Row 3: Beta values
        ax3 = fig.add_subplot(gs[2, i])
        x_pos = np.arange(n_regressors)
        ax3.bar(x_pos, betas[vox_idx, :], alpha=0.7, color='steelblue', label='Estimated')

        if betas_true is not None:
            if torch.is_tensor(betas_true):
                betas_true_np = betas_true.cpu().numpy()
            else:
                betas_true_np = betas_true

            # Handle different shapes
            if betas_true_np.ndim == 1:
                # Same beta for all voxels
                ax3.scatter(x_pos[:len(betas_true_np)], betas_true_np,
                           color='red', s=100, zorder=10, marker='x',
                           linewidths=3, label='True')
            elif betas_true_np.ndim == 2:
                # Different beta per voxel
                ax3.scatter(x_pos, betas_true_np[vox_idx, :],
                           color='red', s=100, zorder=10, marker='x',
                           linewidths=3, label='True')

        ax3.set_xlabel('Regressor')
        ax3.set_ylabel('Beta')
        ax3.set_title('Parameter Estimates')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Right column: Summary statistics
    # Design matrix
    ax_design = fig.add_subplot(gs[0, -1])
    im = ax_design.imshow(design.T, aspect='auto', cmap='RdBu_r',
                          interpolation='nearest', vmin=-design.std()*3,
                          vmax=design.std()*3)
    ax_design.set_xlabel('Time (TRs)')
    ax_design.set_ylabel('Regressor')
    ax_design.set_title('Design Matrix')
    plt.colorbar(im, ax=ax_design, fraction=0.046, pad=0.04)

    # R² histogram
    ax_r2_hist = fig.add_subplot(gs[1, -1])
    ax_r2_hist.hist(r2, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax_r2_hist.axvline(np.mean(r2), color='red', linestyle='--',
                       linewidth=2, label=f'Mean={np.mean(r2):.3f}')
    ax_r2_hist.axvline(np.median(r2), color='orange', linestyle='--',
                       linewidth=2, label=f'Median={np.median(r2):.3f}')
    ax_r2_hist.set_xlabel('R²')
    ax_r2_hist.set_ylabel('Count')
    ax_r2_hist.set_title(f'R² Distribution (n={n_voxels_total})')
    ax_r2_hist.legend()
    ax_r2_hist.grid(True, alpha=0.3)

    # Summary statistics text
    ax_summary = fig.add_subplot(gs[2, -1])
    ax_summary.axis('off')

    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    summary_text += f"Data Shape: {data.shape}\n"
    summary_text += f"Design: {design.shape[0]} TRs × {design.shape[1]} regressors\n\n"
    summary_text += f"R² Statistics:\n"
    summary_text += f"  Mean:   {np.mean(r2):.4f}\n"
    summary_text += f"  Median: {np.median(r2):.4f}\n"
    summary_text += f"  Std:    {np.std(r2):.4f}\n"
    summary_text += f"  Min:    {np.min(r2):.4f}\n"
    summary_text += f"  Max:    {np.max(r2):.4f}\n\n"

    if betas_true is not None:
        if betas_true_np.ndim == 1:
            beta_error = np.abs(betas[:, :len(betas_true_np)] - betas_true_np[np.newaxis, :])
        else:
            beta_error = np.abs(betas - betas_true_np)
        summary_text += f"Beta Estimation Error:\n"
        summary_text += f"  MAE:  {np.mean(beta_error):.4f}\n"
        summary_text += f"  RMSE: {np.sqrt(np.mean(beta_error**2)):.4f}\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Single Simulation Deep Dive', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_batch_summary(
    results_list: List[Dict],
    metrics: List[str] = ['r2', 'beta_error', 'hrf_recovery'],
    group_by: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
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
    n_sims = len(results_list)

    # Extract data
    r2_means = [r.get('r2_mean', r.get('mean_r2', np.nan)) for r in results_list]
    r2_medians = [r.get('r2_median', r.get('median_r2', np.nan)) for r in results_list]
    r2_stds = [r.get('r2_std', 0) for r in results_list]

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
    if 'r2' in metrics:
        ax = axes[plot_idx]

        if group_by and group_by in results_list[0]:
            # Grouped visualization
            groups = sorted(set(r[group_by] for r in results_list))
            x_pos = np.arange(len(groups))

            means_by_group = []
            stds_by_group = []
            for g in groups:
                group_vals = [r['r2_mean'] for r in results_list if r[group_by] == g]
                means_by_group.append(np.mean(group_vals))
                stds_by_group.append(np.std(group_vals))

            ax.bar(x_pos, means_by_group, yerr=stds_by_group,
                  alpha=0.7, capsize=5, color='steelblue', edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{g:.2f}' if isinstance(g, float) else str(g) for g in groups])
            ax.set_xlabel(group_by.replace('_', ' ').title())
        else:
            # Overall distribution
            ax.hist(r2_means, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.mean(r2_means), color='red', linestyle='--',
                      linewidth=2, label=f'Mean={np.mean(r2_means):.3f}')
            ax.legend()

        ax.set_ylabel('R² (Mean across voxels)')
        ax.set_title(f'R² Distribution\n{n_sims} simulations')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot beta estimation error
    if 'beta_error' in metrics:
        ax = axes[plot_idx]

        beta_maes = [r.get('beta_error_mae', np.nan) for r in results_list]
        beta_rmses = [r.get('beta_error_rmse', np.nan) for r in results_list]

        if not all(np.isnan(beta_maes)):
            if group_by and group_by in results_list[0]:
                groups = sorted(set(r[group_by] for r in results_list))
                x_pos = np.arange(len(groups))

                mae_by_group = []
                rmse_by_group = []
                for g in groups:
                    group_mae = [r['beta_error_mae'] for r in results_list
                               if r[group_by] == g and not np.isnan(r.get('beta_error_mae', np.nan))]
                    group_rmse = [r['beta_error_rmse'] for r in results_list
                                if r[group_by] == g and not np.isnan(r.get('beta_error_rmse', np.nan))]
                    mae_by_group.append(np.mean(group_mae) if group_mae else 0)
                    rmse_by_group.append(np.mean(group_rmse) if group_rmse else 0)

                width = 0.35
                ax.bar(x_pos - width/2, mae_by_group, width, label='MAE',
                      alpha=0.7, color='steelblue', edgecolor='black')
                ax.bar(x_pos + width/2, rmse_by_group, width, label='RMSE',
                      alpha=0.7, color='coral', edgecolor='black')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'{g:.2f}' if isinstance(g, float) else str(g) for g in groups])
                ax.set_xlabel(group_by.replace('_', ' ').title())
            else:
                ax.scatter(beta_maes, beta_rmses, alpha=0.5, s=50)
                ax.plot([0, max(beta_maes)], [0, max(beta_maes)], 'k--', alpha=0.3)
                ax.set_xlabel('MAE')
                ax.set_ylabel('RMSE')

            ax.set_title('Beta Estimation Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Plot HRF recovery quality
    if 'hrf_recovery' in metrics:
        ax = axes[plot_idx]

        hrf_corrs = [r.get('hrf_correlation', np.nan) for r in results_list]
        hrf_rmses = [r.get('hrf_rmse', np.nan) for r in results_list]

        if not all(np.isnan(hrf_corrs)):
            if group_by and group_by in results_list[0]:
                groups = sorted(set(r[group_by] for r in results_list))
                x_pos = np.arange(len(groups))

                corr_by_group = []
                for g in groups:
                    group_corr = [r['hrf_correlation'] for r in results_list
                                if r[group_by] == g and not np.isnan(r.get('hrf_correlation', np.nan))]
                    corr_by_group.append(np.mean(group_corr) if group_corr else 0)

                ax.bar(x_pos, corr_by_group, alpha=0.7, color='forestgreen', edgecolor='black')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'{g:.2f}' if isinstance(g, float) else str(g) for g in groups])
                ax.set_xlabel(group_by.replace('_', ' ').title())
                ax.set_ylim([0, 1])
            else:
                ax.hist(hrf_corrs, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
                ax.axvline(np.mean(hrf_corrs), color='red', linestyle='--',
                          linewidth=2, label=f'Mean={np.mean(hrf_corrs):.3f}')
                ax.legend()

            ax.set_ylabel('Correlation with True HRF')
            ax.set_title('HRF Recovery Quality')
            ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Statistical power curve (if effect_size is available)
    if 'power' in metrics and 'effect_size' in results_list[0]:
        ax = axes[plot_idx]

        effect_sizes = sorted(set(r['effect_size'] for r in results_list))
        r2_thresholds = [0.01, 0.05, 0.10, 0.20]

        for threshold in r2_thresholds:
            power = []
            for es in effect_sizes:
                es_sims = [r for r in results_list if r['effect_size'] == es]
                prop_above = np.mean([r['r2_mean'] > threshold for r in es_sims])
                power.append(prop_above)

            ax.plot(effect_sizes, power, marker='o', linewidth=2,
                   label=f'R²>{threshold}')

        ax.axhline(0.8, color='k', linestyle='--', alpha=0.5, label='80% power')
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Statistical Power')
        ax.set_title('Power Curves')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Batch Simulation Summary ({n_sims} simulations)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_parametric_exploration(
    results_grid: Dict,
    x_var: str = 'beta_ratio',
    y_var: str = 'hrf_type',
    z_var: str = 'noise_level',
    metric: str = 'r2_mean',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
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
        im = ax.imshow(grid_data, aspect='auto', cmap=cmap,
                      interpolation='nearest', origin='lower')

        # Set ticks
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{x:.2f}' if isinstance(x, float) else str(x)
                           for x in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{y:.2f}' if isinstance(y, float) else str(y)
                           for y in y_vals])

        ax.set_xlabel(x_var.replace('_', ' ').title())
        ax.set_ylabel(y_var.replace('_', ' ').title())
        ax.set_title(f'{z_var.replace("_", " ").title()} = {z_val}')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if not np.isnan(grid_data[i, j]):
                    text = ax.text(j, i, f'{grid_data[i, j]:.2f}',
                                 ha="center", va="center", color="white" if grid_data[i, j] < np.nanmean(grid_data) else "black",
                                 fontsize=8)

    # Hide unused subplots
    for i in range(n_z, len(axes)):
        axes[i].axis('off')

    metric_title = metric.replace('_', ' ').title()
    plt.suptitle(f'Parametric Exploration: {metric_title}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_hrf_recovery(
    hrf_estimated: Union[torch.Tensor, np.ndarray],
    hrf_true: Union[torch.Tensor, np.ndarray],
    tr: float = 1.0,
    voxel_selection: str = 'best',
    n_voxels: int = 6,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
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
    correlations = np.array([np.corrcoef(hrf_true_norm, hrf_est_norm[i, :])[0, 1]
                            for i in range(n_voxels_total)])

    # Select voxels
    if voxel_selection == 'best':
        voxel_indices = np.argsort(correlations)[-n_voxels:][::-1]
    elif voxel_selection == 'worst':
        voxel_indices = np.argsort(correlations)[:n_voxels]
    elif voxel_selection == 'median':
        sorted_idx = np.argsort(correlations)
        mid = len(sorted_idx) // 2
        start = max(0, mid - n_voxels // 2)
        voxel_indices = sorted_idx[start:start + n_voxels]
    elif voxel_selection == 'random':
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

        ax.plot(time_axis, hrf_true_norm, 'k--', linewidth=2, label='True HRF')
        ax.plot(time_axis, hrf_est_norm[vox_idx, :], 'r-', linewidth=2, label='Estimated')

        corr = correlations[vox_idx]
        rmse = np.sqrt(np.mean((hrf_true_norm - hrf_est_norm[vox_idx, :])**2))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response (norm.)')
        ax.set_title(f'Voxel {vox_idx}\nr={corr:.3f}, RMSE={rmse:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary: Correlation histogram
    ax_hist = fig.add_subplot(gs[0, -1])
    ax_hist.hist(correlations, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax_hist.axvline(np.mean(correlations), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={np.mean(correlations):.3f}')
    ax_hist.axvline(np.median(correlations), color='orange', linestyle='--',
                   linewidth=2, label=f'Median={np.median(correlations):.3f}')
    ax_hist.set_xlabel('Correlation with True HRF')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(f'HRF Recovery Quality\n(n={n_voxels_total})')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Bottom row: Mean estimated HRF ± std
    ax_mean = fig.add_subplot(gs[1, :n_voxels//2 + 1])

    mean_hrf = np.mean(hrf_est_norm, axis=0)
    std_hrf = np.std(hrf_est_norm, axis=0)

    ax_mean.fill_between(time_axis, mean_hrf - std_hrf, mean_hrf + std_hrf,
                        alpha=0.3, color='steelblue', label='±1 SD')
    ax_mean.plot(time_axis, mean_hrf, 'b-', linewidth=2, label='Mean Estimated')
    ax_mean.plot(time_axis, hrf_true_norm, 'k--', linewidth=2, label='True HRF')

    ax_mean.set_xlabel('Time (s)')
    ax_mean.set_ylabel('Response (normalized)')
    ax_mean.set_title('Mean Estimated HRF Across All Voxels')
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.3)

    # Summary statistics text
    ax_stats = fig.add_subplot(gs[1, n_voxels//2 + 1:])
    ax_stats.axis('off')

    # Calculate peak timing error
    true_peak_idx = np.argmax(np.abs(hrf_true_norm))
    true_peak_time = true_peak_idx * tr

    peak_times = np.array([np.argmax(np.abs(hrf_est_norm[i, :])) * tr
                          for i in range(n_voxels_total)])
    peak_errors = peak_times - true_peak_time

    stats_text = "HRF Recovery Statistics\n" + "="*35 + "\n\n"
    stats_text += f"Correlation:\n"
    stats_text += f"  Mean:   {np.mean(correlations):.4f}\n"
    stats_text += f"  Median: {np.median(correlations):.4f}\n"
    stats_text += f"  Std:    {np.std(correlations):.4f}\n"
    stats_text += f"  Min:    {np.min(correlations):.4f}\n"
    stats_text += f"  Max:    {np.max(correlations):.4f}\n\n"

    rmses = np.array([np.sqrt(np.mean((hrf_true_norm - hrf_est_norm[i, :])**2))
                     for i in range(n_voxels_total)])
    stats_text += f"RMSE:\n"
    stats_text += f"  Mean:   {np.mean(rmses):.4f}\n"
    stats_text += f"  Median: {np.median(rmses):.4f}\n\n"

    stats_text += f"Peak Timing Error (s):\n"
    stats_text += f"  True peak: {true_peak_time:.2f}s\n"
    stats_text += f"  Mean error: {np.mean(peak_errors):.2f}s\n"
    stats_text += f"  Std error:  {np.std(peak_errors):.2f}s\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('HRF Recovery Analysis (FIR Estimation)',
                fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_design_comparison(
    designs: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    tr: float = 1.0,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
):
    """
    Compare multiple design matrices visually

    Parameters
    ----------
    designs : dict
        Dictionary mapping design names to design matrices (nt, n_regressors)
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
            label = labels[reg] if labels and reg < len(labels) else f'Reg {reg}'
            ax_time.plot(time_axis, design[:, reg], label=label, linewidth=1.5)

        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'{design_name}\n(Timecourses)')
        ax_time.legend(fontsize=8, loc='upper right')
        ax_time.grid(True, alpha=0.3)

        # Bottom: Heatmap
        ax_heat = axes[1, i]
        im = ax_heat.imshow(design.T, aspect='auto', cmap='RdBu_r',
                           interpolation='nearest',
                           vmin=-np.std(design)*3, vmax=np.std(design)*3)
        ax_heat.set_xlabel('Time (TRs)')
        ax_heat.set_ylabel('Regressor')
        ax_heat.set_title(f'{design_name}\n(Heatmap)')
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    plt.suptitle('Design Matrix Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_interactive_summary_html(
    results_list: List[Dict],
    output_path: str = 'simulation_summary.html'
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
    r2_means = [r.get('r2_mean', r.get('mean_r2', 0)) for r in results_list]

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
        html_content += f"<tr><td>{i+1}</td>"
        for key in keys:
            value = result.get(key, '')
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
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Created interactive HTML summary: {output_path}")
    return output_path
