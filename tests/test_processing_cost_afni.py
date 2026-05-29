"""Tests for the AFNI-faithful cost functions (blok local Pearson + histogram).

These verify the cost *behaviour* (sign, monotonicity, differentiability) on
synthetic data, and — when a local 3dAllineate is available — that the values
match ``3dAllineate -allcostX`` within tolerance.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import cost_hist as ch
from fastfuncstuff.processing.cost_blok import (
    assign_bloks,
    auto_blok_radius,
    local_pearson_value,
    local_pearson_value_batched,
    lpa_cost,
    lpc_cost,
)

DEV = torch.device("cpu")


def _structured(shape=(40, 40, 40), seed=0):
    """A smooth-ish structured volume (blobs + noise)."""
    torch.manual_seed(seed)
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")

    def blob(c, r, a):
        d2 = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2).float()
        return a * torch.exp(-d2 / (2 * r * r))

    v = blob((nx / 2, ny / 2, nz / 2), 8, 100) + blob((nx / 3, ny / 2, nz / 2), 5, 60)
    return v + torch.randn(shape) * 1.0


# ---------------------------------------------------------------------------
# Blok geometry
# ---------------------------------------------------------------------------


class TestBlokGeometry:
    def test_auto_radius_matches_afni_volfac(self):
        # 1 mm iso: rhdd ~6.52, tohd ~5.18 (3dAllineate's 555-voxel target).
        assert auto_blok_radius((1, 1, 1), "rhdd") == pytest.approx(6.52, abs=0.05)
        assert auto_blok_radius((1, 1, 1), "tohd") == pytest.approx(5.18, abs=0.05)

    def test_space_filling_partition(self):
        # Almost every voxel lands in exactly one blok (clean Voronoi tiling).
        bs = assign_bloks((40, 40, 40), (1, 1, 1), "tohd")
        assigned = int((bs.index >= 0).sum())
        assert assigned > 0.8 * bs.index.numel()


# ---------------------------------------------------------------------------
# Local Pearson (lpc / lpa) behaviour
# ---------------------------------------------------------------------------


class TestLocalPearson:
    def test_identical_anti_independent(self):
        base = _structured()
        bs = assign_bloks(base.shape, (1, 1, 1), "tohd")
        v_same = float(local_pearson_value(base, base, None, bs))
        v_anti = float(local_pearson_value(base, -base, None, bs))
        v_indep = float(local_pearson_value(base, torch.randn_like(base), None, bs))
        assert v_same > 1.0  # strong positive local corr
        assert v_anti == pytest.approx(-v_same, rel=1e-4)  # sign-preserving
        assert abs(v_indep) < 0.1  # ~0 for independent

    def test_lpc_lpa_conventions(self):
        base = _structured()
        bs = assign_bloks(base.shape, (1, 1, 1), "tohd")
        val = local_pearson_value(base, base, None, bs)
        # ffs convention (higher == better): lpc = -value, lpa = |value|.
        assert float(lpc_cost(base, base, None, bs)) == pytest.approx(-float(val), rel=1e-5)
        assert float(lpa_cost(base, base, None, bs)) == pytest.approx(abs(float(val)), rel=1e-5)

    def test_batched_matches_serial(self):
        base = _structured()
        bs = assign_bloks(base.shape, (1, 1, 1), "tohd")
        warps = torch.stack([base, -base, torch.randn_like(base)])
        batched = local_pearson_value_batched(base, warps, None, bs)
        for i in range(3):
            serial = local_pearson_value(base, warps[i], None, bs)
            assert float(batched[i]) == pytest.approx(float(serial), rel=1e-4, abs=1e-4)

    def test_differentiable(self):
        base = _structured((24, 24, 24))
        bs = assign_bloks(base.shape, (1, 1, 1), "tohd")
        y = (base * 0.8 + 0.2 * torch.randn_like(base)).requires_grad_(True)
        lpc_cost(base, y, None, bs).backward()
        assert y.grad is not None and bool(torch.isfinite(y.grad).all())


# ---------------------------------------------------------------------------
# Histogram costs
# ---------------------------------------------------------------------------


class TestHistogramCosts:
    def test_nbin_formula(self):
        assert ch.compute_nbin(110592) == 48  # 110592 ** (1/3)
        assert ch.compute_nbin(10) == 5  # clamped low
        assert ch.compute_nbin(10**9) == 255  # clamped high

    def test_mi_higher_for_dependent(self):
        base = _structured()
        mi_dep = float(ch.mi_cost(base, base * 2 + 3))  # affine -> high MI
        mi_indep = float(ch.mi_cost(base, torch.randn_like(base)))
        assert mi_dep > mi_indep

    def test_costs_differentiable(self):
        base = _structured((24, 24, 24))
        for fn in (
            ch.mi_cost,
            ch.nmi_cost,
            ch.je_cost,
            ch.hel_cost,
            lambda b, w: ch.cr_cost(b, w, mode="u"),
        ):
            y = (base * 0.8 + 0.2 * torch.randn_like(base)).requires_grad_(True)
            c = fn(base, y)
            c.backward()
            assert y.grad is not None and bool(torch.isfinite(y.grad).all())

    def test_hellinger_rewards_dependence(self):
        # hel_cost = 1 - affinity; larger when the pair is dependent.
        base = _structured()
        hel_dep = float(ch.hel_cost(base, base * 2 + 1))
        hel_indep = float(ch.hel_cost(base, torch.randn_like(base)))
        assert hel_dep > hel_indep


# ---------------------------------------------------------------------------
# Numeric agreement with 3dAllineate -allcostX (skipped if AFNI absent)
# ---------------------------------------------------------------------------


def _afni_allcostx(base_path, source_path, weight_path):
    """Run 3dAllineate -allcostX -> {name: value}.

    A uniform weight + ``-nmatch 100%`` makes AFNI use every voxel with equal
    weight, matching our unweighted measures (this is the regime where the
    histogram costs agree tightly; AFNI's default autoweight/decimation would
    change the effective sample set).
    """
    out = subprocess.run(
        [
            "3dAllineate",
            "-base",
            base_path,
            "-source",
            source_path,
            "-weight",
            weight_path,
            "-nmatch",
            "100%",
            "-allcostX",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    ).stderr
    vals = {}
    for line in out.splitlines():
        parts = line.split("=")
        if len(parts) == 2 and parts[0].strip() in (
            "ls",
            "mi",
            "nmi",
            "je",
            "hel",
            "crU",
            "crA",
            "crM",
            "lpc",
            "lpa",
        ):
            try:
                vals[parts[0].strip()] = float(parts[1])
            except ValueError:
                pass
    return vals


@pytest.mark.skipif(shutil.which("3dAllineate") is None, reason="3dAllineate not installed")
def test_histogram_costs_match_afni(tmp_path):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    n = 48
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)

    def blob(c, r, a):
        return a * np.exp(-(((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) / (2 * r * r)))

    base = (
        blob((24, 24, 24), 9, 100) + blob((16, 30, 20), 5, 60) + rng.normal(0, 1, (n, n, n))
    ).astype(np.float32)
    src = (np.sqrt(np.abs(base)) * 8 + rng.normal(0, 1, base.shape)).astype(np.float32)
    aff = np.eye(4, dtype=np.float32)
    bp = str(tmp_path / "base.nii")
    sp = str(tmp_path / "src.nii")
    wp = str(tmp_path / "ones.nii")
    nib.save(nib.Nifti1Image(base, aff), bp)
    nib.save(nib.Nifti1Image(src, aff), sp)
    nib.save(nib.Nifti1Image(np.ones_like(base), aff), wp)

    afni = _afni_allcostx(bp, sp, wp)
    if not afni:
        pytest.skip("could not parse 3dAllineate output")

    b = torch.from_numpy(np.ascontiguousarray(base.transpose(2, 1, 0)))
    s = torch.from_numpy(np.ascontiguousarray(src.transpose(2, 1, 0)))
    m = ch.hist2d_measures(b, s)

    # nmi and the correlation-ratio family are global (no blok pruning) -> tight.
    assert m.nmi == pytest.approx(afni["nmi"], abs=0.02)
    assert m.cr_yx == pytest.approx(afni["crU"], abs=0.02)
    # AFNI prints hel = -(1 - affinity); our m.hel is the affinity.
    assert m.hel == pytest.approx(1.0 + afni["hel"], abs=0.02)
