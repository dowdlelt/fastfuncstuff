"""Optimization for warp parameters.

AFNI uses Powell's NEWUOA (derivative-free, quadratic-model trust-region)
optimizer. We use scipy's Powell method with properly tuned parameters,
plus a batched Adam optimizer for GPU-parallel patch processing.

Key fixes vs initial version:
  - maxfev matches AFNI's PRED10 formula: (n+4)*(n+3)//2 + 10
  - Initial directions scaled to param_max (not unit vectors)
  - Proper ftol relative to cost magnitude
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor


def optimize_warp_params_torch(
    cost_fn: Callable[[Tensor], float],
    n_params: int,
    param_max: float,
    device: torch.device,
    max_iter: int = 200,
    prad: float = 0.333,
    tolerance: float = 0.003,
) -> tuple[Tensor, float]:
    """Optimize warp parameters using derivative-free optimization.

    Uses scipy Powell with properly scaled initial directions and generous
    function evaluation budget matching AFNI's NEWUOA settings.

    Args:
        cost_fn: Function mapping parameter tensor (n_params,) -> scalar cost.
        n_params: Number of parameters.
        param_max: Maximum absolute value for each parameter.
        device: Torch device.
        max_iter: Maximum optimization iterations.
        prad: Initial search radius (fraction of param_max).
        tolerance: Convergence tolerance (relative).

    Returns:
        (best_params, best_cost): Optimized parameters and final cost.
    """
    params = torch.zeros(n_params, dtype=torch.float32, device=device)
    best_cost = cost_fn(params)
    best_params = params.clone()

    try:
        from scipy.optimize import minimize as scipy_minimize

        def scipy_cost(x: np.ndarray) -> float:
            p = torch.from_numpy(x.astype(np.float32)).to(device)
            return cost_fn(p)

        x0 = np.zeros(n_params, dtype=np.float64)

        # Match AFNI's PRED10 formula for function evaluation budget:
        #   PRED10(n) = (n+4)*(n+3)/2 + 10
        # NEWUOA uses this as iteration count where each iter is ~1-3 evals.
        # For Powell, each "iteration" (full direction sweep) costs ~n evals,
        # so we multiply by 3 to compensate.
        afni_budget = (n_params + 4) * (n_params + 3) // 2 + 10
        maxfev = afni_budget * 3

        # Initial search directions scaled to param_max
        # Powell's default unit vectors are way too large for our param range
        initial_step = prad * param_max
        direc = np.eye(n_params, dtype=np.float64) * initial_step

        # ftol: relative tolerance on cost improvement
        ftol = tolerance * max(abs(best_cost), 1e-6)

        result = scipy_minimize(
            scipy_cost, x0,
            method='Powell',
            options={
                'maxiter': max_iter,
                'maxfev': maxfev,
                'ftol': ftol,
                'direc': direc,
            },
        )

        final_params = torch.from_numpy(
            np.clip(result.x, -param_max, param_max).astype(np.float32)
        ).to(device)
        final_cost = cost_fn(final_params)

        if final_cost < best_cost:
            return final_params, final_cost
        else:
            return best_params, best_cost

    except ImportError:
        # Fallback: coordinate descent with quadratic interpolation
        return _coordinate_descent(
            cost_fn, n_params, param_max, device, max_iter, prad
        )


def _coordinate_descent(
    cost_fn: Callable[[Tensor], float],
    n_params: int,
    param_max: float,
    device: torch.device,
    max_iter: int,
    prad: float,
) -> tuple[Tensor, float]:
    """Coordinate descent optimizer with quadratic interpolation step."""
    params = torch.zeros(n_params, dtype=torch.float32, device=device)
    best_cost = cost_fn(params)
    radius = prad * param_max

    for _iter in range(max_iter):
        improved = False
        for i in range(n_params):
            orig = params[i].item()

            # Evaluate at +radius
            params[i] = max(-param_max, min(param_max, orig + radius))
            cost_p = cost_fn(params)

            if cost_p < best_cost:
                best_cost = cost_p
                improved = True
                continue

            # Evaluate at -radius
            params[i] = max(-param_max, min(param_max, orig - radius))
            cost_m = cost_fn(params)

            if cost_m < best_cost:
                best_cost = cost_m
                improved = True
                continue

            # Neither worked, restore
            params[i] = orig

        if not improved:
            radius *= 0.5
            if radius < 0.001 * param_max:
                break

    return params, best_cost


def optimize_warp_params_batched(
    batched_cost_fn: Callable[[Tensor], Tensor],
    B: int,
    n_params: int,
    param_max: float,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.005,
    tolerance: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Optimize parameters for B patches simultaneously using autograd.

    All patches are optimized in parallel via Adam with box constraints,
    cosine annealing LR schedule, and gradient clipping.

    Args:
        batched_cost_fn: Maps (B, n_params) tensor -> (B,) cost tensor.
        B: Number of patches.
        n_params: Parameters per patch.
        param_max: Box constraint on each parameter.
        device: Torch device.
        max_iter: Maximum optimizer steps.
        lr: Learning rate for Adam.
        tolerance: Early stopping tolerance.

    Returns:
        (best_params, best_costs): (B, n_params) and (B,) tensors.
    """
    params = torch.zeros(B, n_params, device=device, dtype=torch.float32,
                         requires_grad=True)

    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)

    # Avoid torch.no_grad() for initial eval — if building-block functions
    # are compiled, the grad_mode toggle triggers a dynamo recompilation.
    # params.detach() already prevents gradient tracking.
    best_costs = batched_cost_fn(params.detach()).detach()
    best_params = params.detach().clone()
    prev_total = best_costs.sum().item()

    no_improve_count = 0

    for _step in range(max_iter):
        optimizer.zero_grad(set_to_none=True)

        costs = batched_cost_fn(params)  # (B,) differentiable
        loss = costs.sum()
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Project back into box constraints
        with torch.no_grad():
            params.data.clamp_(-param_max, param_max)

        current_total = loss.item()

        # Per-patch best tracking (no .any() — avoids GPU→CPU sync;
        # torch.where is a no-op when improved is all-False)
        with torch.no_grad():
            improved = costs < best_costs
            best_costs = torch.where(improved, costs.detach(), best_costs)
            best_params = torch.where(improved.unsqueeze(1), params.detach(), best_params)

        # Early stopping on total improvement
        if abs(current_total - prev_total) < tolerance * max(abs(prev_total), 1e-6):
            no_improve_count += 1
            if no_improve_count >= 5:
                break
        else:
            no_improve_count = 0
        prev_total = current_total

    return best_params, best_costs
