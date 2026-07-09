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
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor


class BatchOptStats(NamedTuple):
    """Optimizer-budget telemetry for one batched-patch optimization call."""

    steps_run: int       # Adam steps actually executed (<= max_iter)
    n_patches: int       # number of patches in the batch
    hit_budget: int      # patches still improving when the loop hit max_iter


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
    patience: int = 5,
    clip_group_size: int | None = None,
) -> tuple[Tensor, Tensor, BatchOptStats]:
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
        tolerance: Per-patch relative-improvement threshold for "stalled".
        patience: Consecutive stalled steps before a patch is considered
            converged. The loop stops only once ALL patches have converged.
        clip_group_size: If set, clip the gradient norm independently over
            consecutive groups of this many rows (to ``max_norm=1.0`` each),
            instead of one global norm over the whole batch. Used by the
            source-batched qwarp so a batch of ``N`` volumes clips exactly as
            the ``N`` single-volume runs would (each volume is its own group);
            without it, stacking volumes inflates the shared norm and shrinks
            every step. ``None`` (default) keeps the original global clip so the
            single-volume path is byte-for-byte unchanged.

    Returns:
        (best_params, best_costs, stats): (B, n_params), (B,), and budget stats.
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
    prev_costs = best_costs.clone()

    # Per-patch early stopping: each patch tracks its OWN no-improvement streak,
    # and the loop only stops once EVERY patch has stalled (or the budget is
    # hit). The old code stopped on the *summed* cost, so once the easy patches
    # plateaued the whole batch broke -- starving the few stubborn patches that
    # were still improving (a real under-warp source). Gradients are per-patch
    # independent (loss is a sum of independent per-patch costs), so keeping
    # converged patches in the loop costs nothing, and best_params tracking
    # protects them from late drift.
    no_improve = torch.zeros(B, dtype=torch.int32, device=device)

    steps_run = 0
    for _step in range(max_iter):
        steps_run += 1
        optimizer.zero_grad(set_to_none=True)

        costs = batched_cost_fn(params)  # (B,) differentiable
        loss = costs.sum()
        loss.backward()

        # Gradient clipping to prevent explosion. Grouped clipping keeps each
        # group's step size identical to running that group as its own batch
        # (matches clip_grad_norm_'s max_norm/(norm+1e-6) rescale, per group).
        if clip_group_size is not None and clip_group_size > 0 and params.grad is not None:
            with torch.no_grad():
                g = params.grad.view(B // clip_group_size, clip_group_size * n_params)
                norms = g.norm(dim=1, keepdim=True)
                g.mul_((1.0 / (norms + 1e-6)).clamp(max=1.0))
        else:
            torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # Project back into box constraints
            params.data.clamp_(-param_max, param_max)

            # Per-patch best tracking (torch.where is a no-op when all-False)
            improved = costs < best_costs
            best_costs = torch.where(improved, costs.detach(), best_costs)
            best_params = torch.where(improved.unsqueeze(1), params.detach(), best_params)

            # Per-patch relative improvement vs the previous step
            rel = (prev_costs - costs).abs() / prev_costs.abs().clamp_min(1e-6)
            stalled = rel < tolerance
            no_improve = torch.where(stalled, no_improve + 1, torch.zeros_like(no_improve))
            prev_costs = costs.detach().clone()

            # One GPU→CPU sync per step (same cadence as the old loss.item())
            if bool((no_improve >= patience).all()):
                break

    # Patches still improving (not yet stalled) when we ran out of budget.
    hit_budget = (
        int((no_improve < patience).sum().item()) if steps_run >= max_iter else 0
    )
    return best_params, best_costs, BatchOptStats(steps_run, B, hit_budget)
