"""PathFinder must not duplicate the per-run constant.

``max_poly_degree=0`` does not mean "no polynomials" -- degree 0 IS the
constant, one per run.  PathFinder builds its own block-diagonal nuisance with
per-run Legendre blocks (polort >= 0, so a constant is already in there) and
then asked fit_glm for degree 0 on top, making the design rank deficient.  The
only symptom is a pinv fallback and non-identifiable nuisance coefficients.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from fastfuncstuff.glm.core import construct_polynomial_matrix, fit_glm

PATHFINDER = Path(__file__).resolve().parents[1] / "fastfuncstuff" / "cli" / "pathfinder.py"


def _pathfinder_like_design(n_runs=2, n_tp=40, polort=1, device=torch.device("cpu")):
    """A task column plus PathFinder's own block-diagonal polynomial nuisance."""
    torch.manual_seed(0)
    task = torch.rand(n_runs * n_tp, 1, device=device)
    blocks = [construct_polynomial_matrix(n_tp, polort, device) for _ in range(n_runs)]
    return task, torch.block_diag(*blocks)


def test_degree_zero_on_top_of_block_diag_is_rank_deficient():
    task, nuisance = _pathfinder_like_design()
    n_tp = nuisance.shape[0] // 2

    # What PathFinder used to ask for: fit_glm adds one more constant per run.
    extra = torch.block_diag(*[construct_polynomial_matrix(n_tp, 0, torch.device("cpu"))] * 2)
    duplicated = torch.cat([task, nuisance, extra], dim=1)
    assert torch.linalg.matrix_rank(duplicated.double()) < duplicated.shape[1]

    # What it asks for now (-1 adds nothing).
    clean = torch.cat([task, nuisance], dim=1)
    assert torch.linalg.matrix_rank(clean.double()) == clean.shape[1]


def test_fit_glm_recovers_the_beta_with_polynomials_disabled():
    device = torch.device("cpu")
    task, nuisance = _pathfinder_like_design(device=device)
    design = torch.cat([task, nuisance], dim=1)
    beta_true = 3.0
    data = (design[:, :1] * beta_true).T.repeat(5, 1)

    results = fit_glm(
        data=data,
        design=design,
        tr=2.0,
        max_poly_degree=-1,
        device=device,
        verbose=False,
        task_indices=[0],
    )
    assert torch.allclose(results.betas[:, 0], torch.full((5,), beta_true), atol=1e-3)


def test_pathfinder_never_asks_for_degree_zero():
    tree = ast.parse(PATHFINDER.read_text())
    offenders = [
        kw.value.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "max_poly_degree"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value == 0
    ]
    assert not offenders, (
        f"pathfinder.py lines {offenders} pass max_poly_degree=0 on a design that "
        "already carries per-run constants; -1 is the disabling value"
    )
