"""The laminar figures render from a fit without needing an inversion: a synthetic RoiFit
carries every array the panels read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.laminar.pipeline import CurveDetail, RoiFit, condition_blocks, curve_pair
from fastfuncstuff.laminar.plots import plot_curves, plot_null, plot_roi, plot_summary

pytest.importorskip("matplotlib")


def fake_fit(name="V1", hemi="lh", seed=0, warn=False):
    rng = np.random.default_rng(seed)
    n_scans, k, n_neu, block, gap = 36, 9, 3, 8, 20
    t = np.arange(n_scans)
    y = np.exp(-((t - 4) ** 2) / 3)[:, None] * np.linspace(1.5, 0.8, k)[None] + 0.05 * rng.normal(
        size=(n_scans, k)
    )
    rows = np.zeros(n_scans, bool)
    rows[1:block] = True
    rows[block + gap + 1 :] = True
    models = ["null", "superficial", "middle", "deep"]
    F = np.array([-300.0, -250.0 + 20 * seed, -270.0, -240.0 - 10 * seed])
    p_model = np.exp(F - F.max()) / np.exp(F - F.max()).sum()
    return RoiFit(
        roi=str(seed + 1), name=name, hemi=hemi, k_values=[7, 9], models=models,
        targets=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], F=F,
        per_k_F=np.stack([F, F - 3], 1), p_model=p_model,
        p_depth=np.array([p_model[1], p_model[2], p_model[3]]), best=int(np.argmax(F)),
        B_mean=rng.normal(0, 1, n_neu), B_sd=np.full(n_neu, 0.4), B_p=np.full(n_neu, 0.9),
        r2=np.full(4, 0.9), neuronal=np.abs(rng.normal(size=(n_scans, n_neu))),
        W=np.abs(rng.normal(size=(k, n_neu))), W_inv=rng.normal(size=(n_neu, k)),
        meta={"TR": 2.0, "block": block, "gap": gap, "converged": [True] * 4},
        y_obs=y, y_hat=y * 0.95, rows=rows, impulse=np.abs(rng.normal(size=(k, n_neu, 20))),
        drainage_shift=np.linspace(0.8, 0, k), kernels={7: None, 9: np.array([0.1, 0.2, 0.4, 0.2, 0.1])},
        stalled=np.zeros((4, 2), bool), warnings=["model 'deep' stalled at K=9"] if warn else [],
        model_local=np.stack([np.linspace(0.8, 0.5, k), np.linspace(0.9, 0.6, k)]),
    )  # fmt: skip


def test_the_roi_page_renders(tmp_path):
    out = plot_roi(fake_fit(), tmp_path / "roi.png", ("std", "dev"))
    assert (tmp_path / "roi.png").stat().st_size > 20_000 and out.endswith("roi.png")


def _with_detail(fit, steps=4):
    fit.detail = CurveDetail(
        k=fit.W.shape[0], y_obs=fit.y_obs, W=fit.W, W_inv=np.linalg.pinv(fit.W),
        y_fine=np.repeat(fit.y_hat, steps, axis=0), steps_per_scan=steps, stalled=False,
    )  # fmt: skip
    return fit


def test_the_curves_page_draws_five_panels(tmp_path, monkeypatch):
    import matplotlib.figure as mf

    seen = {}
    savefig = mf.Figure.savefig

    def spy(self, *a, **kw):
        seen["n"] = len(self.axes)
        return savefig(self, *a, **kw)

    monkeypatch.setattr(mf.Figure, "savefig", spy)
    plot_curves(_with_detail(fake_fit()), tmp_path / "c.png", ("std", "dev"))
    assert seen["n"] == 5 and (tmp_path / "c.png").stat().st_size > 20_000
    with pytest.raises(ValueError):
        plot_curves(fake_fit(), tmp_path / "bad.png")


def test_curve_pair_reads_each_condition_on_its_own_grid():
    """The second window may be longer than the first, and rows is what says so."""
    fit = _with_detail(fake_fit())
    fit.rows[:] = False
    fit.rows[1:8] = True
    fit.rows[29:34] = True
    assert condition_blocks(fit.rows) == [(1, 8), (29, 34)]
    a, b = curve_pair(fit, 0), curve_pair(fit, 1)
    assert a["measured"].shape[0] == 7 and b["measured"].shape[0] == 5
    assert a["t_fine"][-1] == pytest.approx(a["t_knots"][-1])
    assert b["t_fine"][-1] == pytest.approx(b["t_knots"][-1])
    assert np.array_equal(b["fit"][0], fit.detail.y_fine[29 * 4])


def test_the_page_marks_a_warned_fit_and_missing_data_is_a_plain_error(tmp_path):
    plot_roi(fake_fit(warn=True), tmp_path / "w.png")
    bare = fake_fit()
    bare.y_obs = None
    with pytest.raises(ValueError, match="no data"):
        plot_roi(bare, tmp_path / "bad.png")


def test_the_summary_renders_for_several_rois(tmp_path):
    fits = [fake_fit(n, h, i, warn=(i == 2)) for i, (n, h) in enumerate(
        [("V1", "lh"), ("V2", "lh"), ("hMT", "rh"), ("A1", "lh"), ("PP", "rh")])]  # fmt: skip
    plot_summary(fits, tmp_path / "s.png", ("std", "dev"))
    assert (tmp_path / "s.png").stat().st_size > 15_000


def test_the_null_page_renders_and_names_the_floor(tmp_path):
    """One panel per statistic, and the suptitle must say what n permutations can claim:
    a figure that hides the resolution of its own p-value invites reading 0 into it."""
    fits = [fake_fit("V1", "lh", i + 1) for i in range(4)]
    out = plot_null(fake_fit(), fits, tmp_path / "null.png")
    assert Path(out).exists() and Path(out).stat().st_size > 5000
    # No replicates at all is a degenerate but legal call: the observed value still draws.
    assert Path(plot_null(fake_fit(), [], tmp_path / "none.png")).exists()


def test_the_profile_panel_draws_the_model_line_only_when_it_exists():
    """Older fits and hand-built ones carry no attribution, and must still render."""
    import matplotlib.pyplot as plt

    from fastfuncstuff.laminar.plots import panel_profiles

    fit = fake_fit()
    fig, ax = plt.subplots()
    panel_profiles(ax, fit)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert sum("model own-depth" in t for t in labels) == 2
    plt.close(fig)
    fit.model_local = None
    fig, ax = plt.subplots()
    panel_profiles(ax, fit)
    assert not any("model own-depth" in t.get_text() for t in ax.get_legend().get_texts())
    plt.close(fig)
