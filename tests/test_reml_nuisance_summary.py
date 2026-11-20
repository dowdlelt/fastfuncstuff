"""ffs_reml -nuisance_summary: per group, then per column, size first and significance last.

Synthetic run: a task in every voxel, motion injected into the first half only, and an
ffs_physio-style slibase file (ColumnLabels s000.card.c1 ...) whose cardiac pair is
injected into the last slice only. The maps must light up where each was put, the
groups must follow the labels, and -events designs (which keep only task betas) must
still report the nuisance coefficients.
"""

from __future__ import annotations

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.glm.nuisance_summary import build_nuisance_sets, group_name
from fastfuncstuff.io.headers import read_brick_labels

SHAPE, T, TR = (6, 5, 4), 120, 2.0


def test_default_groups_follow_the_labels():
    labels = ["task#0", "Run#1Pol#0", "Run#1Pol#1", "mot_00", "mot_01", "mot_deriv_00",
              "card.c1", "card.s1", "resp.c1"]  # fmt: skip
    groups, sets = build_nuisance_sets(labels, [0], ["dsort0"])
    assert [g.name for g in groups] == ["polort", "mot", "mot_deriv", "card", "resp", "dsort"]
    assert groups[-1].columns == [len(labels)]  # dsort column k = n_design + k
    # drift without the run mean (Pol#0); group entry only
    assert not groups[0].per_column and sets[groups[0].column_sets[0]] == [2]
    # each group, then its columns alone
    assert [sets[i] for i in groups[1].column_sets] == [[3, 4], [3], [4]]
    over, _ = build_nuisance_sets(labels, [0], overrides=[("motion", r"^mot")])
    assert [g.name for g in over][:2] == ["polort", "motion"] and len(over[1].columns) == 3
    assert group_name("mot_demean.r02[4]") == "mot_demean"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("nsum")
    rng = np.random.default_rng(0)
    t = np.arange(T) * TR
    box = ((t % 40) < 20).astype(float)
    x = np.convolve(box, np.exp(-((np.arange(0, 20, TR) - 6) ** 2) / 8))[:T]
    x /= x.max()
    mot = np.cumsum(rng.standard_normal((T, 3)), 0) * 0.3
    mot -= mot.mean(0)
    deriv = np.vstack([np.zeros((1, 3)), np.diff(mot, axis=0)])
    nz = SHAPE[2]
    card = np.stack(
        [
            np.stack([np.cos(2 * np.pi * 1.1 * t + s), np.sin(2 * np.pi * 1.1 * t + s)])
            for s in range(nz)
        ]
    )  # (nz, 2, T), its own phase per slice
    data = 100 + 2.0 * x + 0.02 * t + rng.normal(0, 1, (*SHAPE, T))
    data[:3] += 4.0 * mot[:, 0]  # motion in the first half (x < 3)
    data[..., nz - 1, :] += 3.0 * card[nz - 1, 0]  # cardiac in the last slice
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = TR
    img.to_filename(str(d / "vol.nii"))
    (d / "events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        + "".join(f"{o}\t20\ttask\n" for o in np.arange(0, T * TR, 40))
    )
    np.savetxt(d / "mot.1D", mot, fmt="%.6f")
    np.savetxt(d / "deriv.1D", deriv, fmt="%.6f")
    # slice-major, as ffs_physio writes it: s000.card.c1 s000.card.s1 s001.card.c1 ...
    cols = np.concatenate([card[s].T for s in range(nz)], 1)
    labs = " ; ".join(f"s{s:03d}.card.{c}" for s in range(nz) for c in ("c1", "s1"))
    with open(d / "physio.slibase.1D", "w") as fh:
        fh.write(f'# <RetroTSout\n# ColumnLabels = "{labs}"\n# >\n')
        np.savetxt(fh, cols, fmt="%.6f")
    return d


def _reml(monkeypatch, d, *argv):
    from fastfuncstuff.cli import reml

    monkeypatch.chdir(d)
    monkeypatch.setattr(
        sys, "argv",
        ["ffs_reml", "-input", "vol.nii", "-events", "events.tsv", "-polort", "2",
         "-ortvec", "mot.1D", "mot", "-ortvec", "deriv.1D", "mot_deriv",
         "-slibase_sm", "physio.slibase.1D", *argv, "-device", "cpu", "-verb", "0"],
    )  # fmt: skip
    reml.main()


def test_summary_bucket_order_groups_and_where_each_regressor_worked(world, monkeypatch):
    _reml(monkeypatch, world, "-nuisance_summary", "ns.nii")
    labels = read_brick_labels(nib.load(str(world / "ns.nii")))
    v = np.asarray(nib.load(str(world / "ns.nii")).dataobj, np.float64)
    lab = {name: i for i, name in enumerate(labels)}
    # group first (size, then significance), then its columns (Coef first, Tstat last)
    i = labels.index("mot_PSC")
    assert labels[i : i + 4] == ["mot_PSC", "mot_R2unique", "mot_R2partial", "mot_Fstat"]
    assert labels[i + 4 : i + 9] == [
        "mot_00_Coef", "mot_00_PSC", "mot_00_R2unique", "mot_00_R2partial", "mot_00_Tstat",
    ]  # fmt: skip
    assert "polort_Fstat" in lab and not any(n.startswith("Run#") for n in labels)
    assert v[..., lab["polort_R2unique"]].max() < 1  # the run means are not in it
    assert "card_Fstat" in lab and "card.c1_Coef" in lab
    # motion column 0 was injected into x < 3 at 4.0 per unit
    coef = v[..., lab["mot_00_Coef"]]
    np.testing.assert_allclose(coef[:3].mean(), 4.0, atol=0.15)
    assert abs(coef[3:].mean()) < 0.15
    F = v[..., lab["mot_Fstat"]]
    assert F[:3].min() > 50 and np.median(F[3:]) < 3
    # PSC ~ SD of 4 * mot_0 over a mean of ~100
    psc = v[..., lab["mot_PSC"]]
    np.testing.assert_allclose(
        np.median(psc[:3]), 100 * 4.0 * np.loadtxt(world / "mot.1D")[:, 0].std() / 100, rtol=0.1
    )
    # cardiac only in the last slice
    cF = v[..., lab["card_Fstat"]]
    assert cF[:, :, -1].min() > 50 and np.median(cF[:, :, :-1]) < 3
    # R2partial is consistent with F: q F / (q F + dof)
    r2p = v[..., lab["mot_R2partial"]]
    assert np.all((r2p >= 0) & (r2p <= 1)) and r2p[:3].min() > r2p[3:].max()


def test_nuisance_group_merges_and_needs_the_summary(world, monkeypatch):
    _reml(monkeypatch, world, "-nuisance_summary", "ns2.nii",
          "-nuisance_group", "motion", "^mot")  # fmt: skip
    labels = read_brick_labels(nib.load(str(world / "ns2.nii")))
    assert (
        "motion_Fstat" in labels and "mot_Fstat" not in labels and "mot_deriv_Fstat" not in labels
    )
    with pytest.raises(SystemExit, match="only shapes -nuisance_summary"):
        _reml(monkeypatch, world, "-Rbuck", "x.nii", "-nuisance_group", "m", "mot")
