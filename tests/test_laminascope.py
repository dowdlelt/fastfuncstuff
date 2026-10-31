"""ffs_laminascope: the comparison spec, and the tool end to end on an analytic sphere."""

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.laminar.comparisons import make_comparison, read_comparisons, resolve

LABELS = ["expres_hi", "expres_lo", "fixate_hi", "fixate_lo", "gender_hi", "gender_lo"]


def test_globs_resolve_in_label_order_and_must_match():
    assert resolve(["*_lo"], LABELS, "x") == ["expres_lo", "fixate_lo", "gender_lo"]
    assert resolve(["gender_lo", "gender_*"], LABELS, "x") == ["gender_hi", "gender_lo"]
    with pytest.raises(ValueError, match="matches no condition"):
        resolve(["faces"], LABELS, "x")


def test_a_label_cannot_sit_on_both_sides():
    with pytest.raises(ValueError, match="on both sides"):
        make_comparison("bad", "*_hi", "gender_*", LABELS)


def test_read_comparisons(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[[comparison]]\nname = "gender_vs_fixate"\nbaseline = "fixate_*"\ntest = "gender_*"\n'
        '[[comparison]]\nname = "hi_vs_lo"\nbaseline = ["*_lo"]\ntest = "*_hi"\n'
        'models = ["null", "deep"]\n'
    )
    a, b = read_comparisons(str(p), LABELS)
    assert a.baseline == ["fixate_hi", "fixate_lo"] and a.test == ["gender_hi", "gender_lo"]
    assert b.models == ["null", "deep"] and len(b.baseline) == 3
    p.write_text('[[comparison]]\nname = "x"\nbaseline = "*_lo"\ntest = "*_hi"\ncolour = 1\n')
    with pytest.raises(ValueError, match="unknown key"):
        read_comparisons(str(p), LABELS)


@pytest.mark.slow  # 25 s: two runs (local vascular prior, then priors from the first run)
def test_the_tool_runs_on_a_sphere_and_paints_back(tmp_path):
    from fastfuncstuff.cli import laminascope
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_surface
    from tests.test_surface_voxelize import icosphere

    v, f = icosphere(4)
    rw, rp = 15.0, 21.0
    save_gifti_surface(tmp_path / "lh.white.surf.gii", v * rw, f)
    save_gifti_surface(tmp_path / "lh.pial.surf.gii", v * rp, f)
    n = 26
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ aff[:3, :3].T + aff[:3, 3]
    r = np.linalg.norm(xyz, axis=-1)
    # Responsive cap: z > 0.8 of the sphere; curves an HRF-ish bump, test 30% larger
    # and more so toward pial.
    t = np.arange(8) * 2.0
    hrf = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
    hrf /= hrf.max()
    depth = np.clip((r - rw) / (rp - rw), 0, 1)
    rng = np.random.default_rng(0)
    for lab, gain in (("a_base", 1.0), ("a_test", 1.3)):
        amp = (0.5 + depth) * gain
        cur = amp[..., None] * hrf + 0.01 * rng.standard_normal((n, n, n, t.size))
        nib.save(nib.Nifti1Image(cur.astype(np.float32), aff), tmp_path / f"c_iresp_{lab}.nii.gz")
    resp = ((xyz[..., 2] > 0.8 * r) & (r > rw - 2)).astype(np.float32)
    nib.save(nib.Nifti1Image(resp, aff), tmp_path / "resp.nii.gz")
    prefix = str(tmp_path / "out" / "s")
    (tmp_path / "out").mkdir()
    rc = laminascope.main([
        "-iresp_prefix", str(tmp_path / "c"), "-window", "0", "14", "-duration", "1",
        "-compare", "ab", "a_base", "a_test",
        "-white_mesh", str(tmp_path / "lh.white.surf.gii"),
        "-pial_mesh", str(tmp_path / "lh.pial.surf.gii"),
        "-responsive", str(tmp_path / "resp.nii.gz"), "-column_radius", "8",
        "-column_spacing", "6", "-depths", "6", "-models", "null", "superficial",
        "-max_iter", "3", "-rescue", "0", "-vascular_prior", "local", "-max_columns", "3", "-device", "cpu",
        "-prefix", prefix, "-verb", "0",
    ])  # fmt: skip
    assert rc == 0
    data, meta = load_gifti_data(f"{prefix}_ab.lh.func.gii")
    names = meta["maps"].split(",")
    assert data.shape == (len(v), len(names)) and "agreement" in names
    covered = data[:, names.index("n_columns")]
    assert np.nanmax(covered) >= 1 and np.isnan(covered).any()  # painted where columns are
    p_sup = data[:, names.index("p_superficial")]
    ok = np.isfinite(p_sup)
    assert ok.any() and ((p_sup[ok] >= 0) & (p_sup[ok] <= 1)).all()
    rows = open(f"{prefix}_ab_columns.tsv").read().splitlines()
    assert len(rows) == 4 and rows[0].startswith("column\themi")
    z = np.load(f"{prefix}_columns.npz")
    assert z["ab_p_model"].shape == (3, 2) and z["psf_K6"].shape[1:] == (6, 6)
    vol = nib.load(f"{prefix}_ab_ribbon.nii.gz")
    assert vol.shape[:3] == (n, n, n)
    assert z["ab_post_s_d_mean"].shape == (3,) and (z["ab_post_s_d_sd"] > 0).all()

    # The same columns again, priors on the vascular parameters from the first run.
    second = str(tmp_path / "out" / "t")
    rc = laminascope.main([
        "-iresp_prefix", str(tmp_path / "c"), "-window", "0", "14", "-duration", "1",
        "-compare", "ab", "a_base", "a_test",
        "-white_mesh", str(tmp_path / "lh.white.surf.gii"),
        "-pial_mesh", str(tmp_path / "lh.pial.surf.gii"),
        "-responsive", str(tmp_path / "resp.nii.gz"), "-column_radius", "8",
        "-column_spacing", "6", "-depths", "6", "-models", "null", "superficial",
        "-max_iter", "3", "-rescue", "0", "-max_columns", "3", "-device", "cpu",
        "-vascular_from", f"{prefix}_columns.npz", "ab", "-prefix", second, "-verb", "0",
    ])  # fmt: skip
    assert rc == 0
    z2 = np.load(f"{second}_columns.npz")
    assert np.array_equal(z2["fitted"], z["fitted"])
    # The prior narrowed to the first posterior: the second posterior is no wider.
    assert (z2["ab_post_s_d_sd"] <= z["ab_post_s_d_sd"] + 1e-9).all()
