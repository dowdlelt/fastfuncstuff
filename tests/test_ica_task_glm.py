"""Task GLM per ICA component: does the identification table find the planted signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.decomposition.postprocess import (
    format_component_task_table,
    save_component_task_table,
)
from fastfuncstuff.decomposition.tools import (
    component_condition_correlations,
    component_task_glm,
)


def _design(n_t: int, n_cond: int, seed: int = 0) -> torch.Tensor:
    """Correlated block regressors -- the case where marginal r is capped."""
    rng = np.random.default_rng(seed)
    cols = []
    for c in range(n_cond):
        x = np.zeros(n_t)
        for onset in range(4 + 3 * c, n_t - 12, 24):
            x[onset : onset + 8] = 1.0
        x = np.convolve(x, np.exp(-((np.arange(20) - 5) ** 2) / 8.0), mode="same")
        cols.append(x + 0.01 * rng.standard_normal(n_t))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float64)


def test_planted_component_tops_the_table():
    n_t, n_cond = 200, 3
    design = _design(n_t, n_cond)
    rng = np.random.default_rng(1)
    # comp 0: the whole design. comp 1: condition 2 only. comp 2-4: pure noise.
    mixing = torch.tensor(
        np.stack(
            [
                design.numpy().sum(axis=1) + 0.3 * rng.standard_normal(n_t),
                design.numpy()[:, 1] + 0.3 * rng.standard_normal(n_t),
                rng.standard_normal(n_t),
                rng.standard_normal(n_t),
                rng.standard_normal(n_t),
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )
    labels = [f"cond{c + 1}" for c in range(n_cond)]
    glm = component_task_glm(mixing, design, labels)

    # The identification question: the two task components rank above the noise.
    assert int(np.argmax(glm["r2"])) in (0, 1)
    assert glm["r2"][0] > 0.8 and glm["r2"][1] > 0.8
    assert glm["r2"][2:].max() < 0.2
    assert glm["p_value"][:2].max() < 1e-20
    assert glm["p_value"][2:].min() > 1e-3

    # The joint fit attributes comp 1 to the condition it was built from.
    assert int(np.argmax(np.abs(glm["t"][1]))) == 1

    # Component 0 is an EQUAL sum of all three conditions, so an honest per-condition
    # readout should treat them alike. The joint fit does; the marginal correlation
    # spreads them by how much each condition happens to overlap the others -- design
    # geometry, not response. This is the whole reason the table is a GLM and not the
    # correlation next door.
    marg = component_condition_correlations(mixing, design)
    assert np.ptp(marg[0]) > 4 * np.ptp(glm["r_joint"][0])

    # A noise component sits near the analytic chance level, not at zero.
    assert glm["r2_chance"] > 0
    assert abs(float(np.median(glm["r2"][2:])) - glm["r2_chance"]) < 0.05


def test_constant_component_scores_zero():
    """Constant timecourses scoring a perfect fit is a bug this repo has shipped twice."""
    n_t = 120
    design = _design(n_t, 2)
    mixing = torch.zeros(n_t, 2, dtype=torch.float64)
    mixing[:, 0] = 1.0
    mixing[:, 1] = torch.tensor(np.random.default_rng(2).standard_normal(n_t))
    glm = component_task_glm(mixing, design, ["a", "b"])
    assert glm["r2"][0] == 0.0
    assert np.all(glm["t"][0] == 0.0)


def test_table_and_tsv_round_trip(tmp_path):
    n_t = 150
    design = _design(n_t, 2)
    mixing = torch.tensor(np.random.default_rng(3).standard_normal((n_t, 4)), dtype=torch.float64)
    glm = component_task_glm(
        mixing, design, ["a", "b"], explained_share=np.array([0.4, 0.3, 0.2, 0.1])
    )
    text = format_component_task_table(glm, top_n=2)
    assert "strongest condition" in text and "2 more" in text

    out = save_component_task_table(glm, tmp_path / "t.tsv")
    rows = out.read_text().strip().split("\n")
    assert rows[0].startswith("#")
    header = rows[1].split("\t")
    assert header[:6] == ["component", "var_share", "r_full", "r2", "F", "p"]
    assert "beta_a" in header and "r_marginal_b" in header
    assert len(rows) == 2 + 4
    assert all(len(r.split("\t")) == len(header) for r in rows[2:])


def test_guidance_ranks_on_the_joint_fit_not_the_marginal_ceiling():
    """The guidance score must not rank a component by design overlap.

    Two components at the same noise level: one holds the WHOLE six-condition
    response, the other holds a single condition. The first has more task in it and
    r_full says so -- but its variance is spread over six regressors while the
    second's sits on one, so the largest marginal |r| ranks them backwards.
    """
    from fastfuncstuff.decomposition.workflow import compute_guidance_scores

    n_t, n_cond = 200, 6
    design = _design(n_t, n_cond)
    d = design.numpy()
    rng = np.random.default_rng(7)
    mixing = torch.tensor(
        np.stack(
            [
                d.sum(axis=1) / d.sum(axis=1).std() + rng.standard_normal(n_t),
                d[:, 1] / d[:, 1].std() + rng.standard_normal(n_t),
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )
    labels = [f"cond{c + 1}" for c in range(n_cond)]
    glm = component_task_glm(mixing, design, labels)
    marg = component_condition_correlations(mixing, design)

    # The premise: the whole-design component fits better, yet loses on marginal |r|.
    assert glm["r_full"][0] > glm["r_full"][1]
    assert np.max(np.abs(marg[0])) < np.max(np.abs(marg[1]))

    comp = np.zeros((2, 10), dtype=np.float32)
    kw: dict = dict(
        comp_np=comp,
        z_maps=None,
        ortvec_corr=None,
        guidance_good_masks=[],
        guidance_bad_masks=[],
        depth_mask_info=None,
        good_z_thresh=2.0,
        out_prefix=Path("unused"),
        run_tag="run01",
        run_idx=0,
    )
    marginal = compute_guidance_scores(condition_corr=marg, **kw)
    joint = compute_guidance_scores(condition_corr=marg, condition_glm=glm, **kw)

    assert marginal["temporal_good_scores"][0] < marginal["temporal_good_scores"][1]
    assert joint["temporal_good_scores"][0] > joint["temporal_good_scores"][1]
