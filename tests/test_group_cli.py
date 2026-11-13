"""ffs_group end to end on tiny synthetic files (CPU)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from fastfuncstuff.cli.group import main, parse_set, short_label
from fastfuncstuff.io.afni import save_nifti
from fastfuncstuff.io.headers import read_brick_labels, read_brick_stataux


def _write_sets(tmp_path, na=6, nb=5, shape=(6, 5, 4)):
    rng = np.random.default_rng(0)
    a, b = [], []
    for grp, n, lst in (("a", na, a), ("b", nb, b)):
        for i in range(n):
            p = tmp_path / f"{grp}{i}.nii.gz"
            # Two sub-bricks so selectors are exercised; brick 1 is the data.
            vol = np.stack([np.zeros(shape), rng.standard_normal(shape) + (grp == "a")], -1)
            save_nifti(vol.astype(np.float32), p, affine=np.eye(4), brick_labels=["junk", "beta"])
            lst.append(p)
    return a, b


def test_two_sample_with_label_selector_and_covariates(tmp_path):
    a, b = _write_sets(tmp_path)
    cov = tmp_path / "cov.txt"
    rows = [f"s{k} {k * 1.5:.2f}" for k in range(11)]
    cov.write_text("subj age\n" + "\n".join(rows) + "\nextra 9\n")
    seta = ["A"] + [x for k, p in enumerate(a) for x in (f"s{k}", f"{p}[beta]")]
    setb = ["B"] + [x for k, p in enumerate(b) for x in (f"s{k + 6}", f"{p}[1]")]
    out = tmp_path / "tt.nii.gz"
    main(["-setA", *seta, "-setB", *setb, "-covariates", str(cov), "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    labs = read_brick_labels(img)
    assert labs[:4] == ["A-B_mean", "A-B_Tstat", "A-B_age", "A-B_age_Tstat"]
    assert read_brick_stataux(img)[1] == (3, (7.0,))  # 11 - 2 - 2*1


def test_short_form_one_sample_matches_scipy(tmp_path):
    a, _ = _write_sets(tmp_path)
    out = tmp_path / "one.nii.gz"
    main(["-setA", *[f"{p}[1]" for p in a], "-labelA", "vis", "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    assert read_brick_labels(img) == ["vis_mean", "vis_Tstat"]
    data = np.stack([np.asarray(nib.load(p).dataobj)[..., 1] for p in a])
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[..., 1], stats.ttest_1samp(data, 0).statistic, rtol=1e-4
    )


def test_set_parsing_rules(tmp_path):
    a, _ = _write_sets(tmp_path, na=2, nb=0)
    assert short_label("/x/Elvis_Presley_long.nii.gz[3]") == "Elvis_Presle"
    lab, subj, ds = parse_set([str(p) for p in a], "SetA", None)
    assert (lab, subj) == ("SetA", ["a0", "a1"])
    with pytest.raises(ValueError, match="'s1' has no dataset"):
        parse_set(["G", "s0", str(a[0]), "s1"], "SetA", None)
    with pytest.raises(ValueError, match="drop -labelA"):
        parse_set(["G", "s0", str(a[0])], "SetA", "X")


def test_missing_covariate_row_is_an_error(tmp_path, capsys):
    a, _ = _write_sets(tmp_path, nb=0)
    cov = tmp_path / "cov.txt"
    cov.write_text("subj age\na0 1\n")
    with pytest.raises(SystemExit):
        main(["-setA", *[f"{p}[1]" for p in a], "-covariates", str(cov), "-prefix",
              str(tmp_path / "x"), "-device", "cpu"])  # fmt: skip
    assert "no row for subject" in capsys.readouterr().err


def test_mema_short_form_beta_t_pairs(tmp_path):
    rng = np.random.default_rng(1)
    shape = (5, 4, 3)
    specs = []
    for i in range(7):
        b = rng.normal(0.5, 1.0, shape)
        t = b / rng.uniform(0.3, 1.0, shape)
        p = tmp_path / f"s{i}.nii.gz"
        save_nifti(np.stack([b, t], -1).astype(np.float32), p, affine=np.eye(4),
                   brick_labels=["c#0_Coef", "c#0_Tstat"])  # fmt: skip
        specs += [f"{p}[c#0_Coef]", f"{p}[c#0_Tstat]"]
    out = tmp_path / "mema.nii.gz"
    main(["-model", "mema", "-setA", *specs, "-labelA", "c", "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    assert read_brick_labels(img) == ["c:b", "c:t", "I2", "tau", "QE:Chisq"]
    assert read_brick_stataux(img)[1] == (3, (6.0,))


def test_mema_only_flags_are_refused_for_ols(tmp_path, capsys):
    a, _ = _write_sets(tmp_path, nb=0)
    with pytest.raises(SystemExit):
        main(["-setA", *[f"{p}[1]" for p in a], "-no_HKtest", "-prefix", str(tmp_path / "x"),
              "-device", "cpu"])  # fmt: skip
    assert "-model mema only" in capsys.readouterr().err


def test_clustsim_and_etac_outputs(tmp_path):
    from fastfuncstuff.stats.clustsim import read_clustsim_tables_from

    a, b = _write_sets(tmp_path, na=8, nb=7, shape=(9, 8, 7))
    out = tmp_path / "g.nii.gz"
    main(["-setA", *[f"{p}[1]" for p in a], "-setB", *[f"{p}[1]" for p in b], "-prefix",
          str(out), "-device", "cpu", "-verb", "0", "-clustsim", "-etac", "-etac_blur", "0",
          "3", "-etac_pthr", "0.01", "0.001", "-niter", "200"])  # fmt: skip
    assert (2, "bi-sided") in read_clustsim_tables_from(out)
    # Every non-main t brick gets its own file and its own tables.
    for lab in ("SetA_mean", "SetB_mean"):
        f = tmp_path / f"g.{lab}.nii.gz"
        assert read_brick_labels(nib.load(f)) == [lab, lab.replace("mean", "Tstat")]
        assert (2, "bi-sided") in read_clustsim_tables_from(f)
    # AFNI's ETAC masks, as uint8, in their side folder.
    allm = tmp_path / "g.etac" / "g.ETACmaskALL.global.2sid.5perc.nii.gz"
    assert read_brick_labels(nib.load(allm)) == ["B0.0_p0.01", "B0.0_p0.001", "B3.0_p0.01",
                                                 "B3.0_p0.001"]  # fmt: skip
    assert nib.load(allm).get_data_dtype() == np.uint8
    union = nib.load(tmp_path / "g.etac" / "g.ETACmask.global.2sid.5perc.nii.gz").get_fdata()
    assert (tmp_path / "g.etac" / "g.ETAC.thresholds.json").exists()
    # The result bucket: z, FWE z (a fizt the viewer reads as the FWE p), p, clusters.
    res = nib.load(tmp_path / "g.ETAC.nii.gz")
    assert read_brick_labels(res) == ["Z_B0.0", "ETAC_FWE_Z", "ETAC_FWE_p", "ETAC_clust_5perc"]
    vol = res.get_fdata()
    assert np.array_equal(vol[..., 2] <= 0.05, union > 0)
    assert np.array_equal(vol[..., 3] > 0, union > 0)
    head = (tmp_path / "g.ETAC.clusters.tsv").read_text().splitlines()[0].split("\t")
    assert head[:4] == ["fpr_goal", "cluster", "sign", "n_vox"] and "fwe_p" in head
    # Side files in one folder, none at the top level.
    assert (tmp_path / "g.clustsim" / "g.CSim.NN2_bisided.niml").exists()
    assert (tmp_path / "g.clustsim" / "g.SetA_mean.CSim.NN2_bisided.niml").exists()
    assert not list(tmp_path.glob("*.CSim.*"))


def _write_buckets(tmp_path, n=6, labels=("vis#0_Coef", "vis#0_Tstat"), tstat_aux=True):
    rng = np.random.default_rng(2)
    paths = []
    for i in range(n):
        b = rng.normal(0.5, 1.0, (5, 4, 3))
        t = b / rng.uniform(0.3, 1.0, b.shape)
        p = tmp_path / f"sub{i}_stats.nii.gz"
        aux = {1: (3, (40.0,))} if tstat_aux else None
        save_nifti(np.stack([b, t], -1).astype(np.float32), p, affine=np.eye(4),
                   brick_labels=list(labels), brick_stataux=aux)  # fmt: skip
        paths.append(p)
    return paths


def test_mema_finds_the_tstat_by_label_and_with_brick(tmp_path):
    paths = _write_buckets(tmp_path)
    label, subj, groups = parse_set([str(p) for p in paths], "SetA", None, True, "vis#0_Coef")
    assert subj[0] == "sub0_stats"
    assert groups[0] == [f"{paths[0]}[vis#0_Coef]", f"{paths[0]}[1]"]


def test_mema_falls_back_to_next_brick_only_if_marked_t(tmp_path):
    paths = _write_buckets(tmp_path, labels=("beta", "stat"))
    _, _, groups = parse_set([f"{p}[0]" for p in paths], "SetA", None, True)
    assert groups[0][1] == f"{paths[0]}[1]"
    bare = _write_buckets(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
                          labels=("beta", "stat"), tstat_aux=False)  # fmt: skip
    with pytest.raises(ValueError, match="cannot find its t-stat"):
        parse_set([f"{p}[0]" for p in bare], "SetA", None, True)


def test_mema_refuses_t_bricks_as_betas_and_cross_file_pairs(tmp_path):
    paths = _write_buckets(tmp_path)
    with pytest.raises(ValueError, match="marked as a t statistic"):
        parse_set([f"{p}[1]" for p in paths], "SetA", None, True)
    # Explicit pairs from the same file are honoured in short form.
    items = [x for p in paths for x in (f"{p}[0]", f"{p}[vis#0_Tstat]")]
    _, subj, groups = parse_set(items, "SetA", None, True)
    assert len(groups) == len(paths) and groups[2][1] == f"{paths[2]}[vis#0_Tstat]"


def test_clustsim_blur_cases_get_their_own_files_and_tables(tmp_path):
    from fastfuncstuff.stats.clustsim import read_clustsim_tables_from

    a, _ = _write_sets(tmp_path, na=8, nb=0, shape=(9, 8, 7))
    out = tmp_path / "c.nii.gz"
    main(["-setA", *[f"{p}[1]" for p in a], "-prefix", str(out), "-device", "cpu", "-verb",
          "0", "-clustsim", "-etac_blur", "0", "2", "-niter", "200"])  # fmt: skip
    blurred = tmp_path / "c.B2.0.nii.gz"
    t0 = read_clustsim_tables_from(out)[(2, "bi-sided")]
    t2 = read_clustsim_tables_from(blurred)[(2, "bi-sided")]
    assert read_brick_labels(nib.load(blurred)) == ["SetA_mean", "SetA_Tstat"]
    # Blurring makes null clusters bigger, so the blurred table asks for more voxels.
    assert t2.sizes[0, 1] > t0.sizes[0, 1]


@pytest.mark.parametrize(
    "extra", [["-model", "mema"], ["-model", "mema", "-unpooled"], ["-unpooled"]]
)
def test_clustsim_and_etac_for_every_model(tmp_path, extra):
    from fastfuncstuff.stats.clustsim import read_clustsim_tables_from

    paths = _write_buckets(tmp_path, n=14)
    a, b = paths[:8], paths[8:]
    out = tmp_path / "m.nii.gz"
    main(["-setA", *map(str, a), "-setB", *map(str, b), "-brick", "vis#0_Coef", *extra,
          "-prefix", str(out), "-device", "cpu", "-verb", "0", "-clustsim", "-etac",
          "-etac_pthr", "0.01", "0.002", "-niter", "60"])  # fmt: skip
    assert (2, "bi-sided") in read_clustsim_tables_from(out)
    assert (tmp_path / "m.etac" / "m.ETACmask.global.2sid.5perc.nii.gz").exists()
    assert (tmp_path / "m.ETAC.nii.gz").exists()
    side = sorted(p.name for p in tmp_path.glob("m.*.nii.gz") if "ETAC" not in p.name)
    assert len(side) == 2  # the two per-set tests, each with its own tables
    for name in side:
        assert (2, "bi-sided") in read_clustsim_tables_from(tmp_path / name)


def test_mema_refuses_etac_blur(tmp_path, capsys):
    paths = _write_buckets(tmp_path, n=6)
    with pytest.raises(SystemExit):
        main(["-model", "mema", "-setA", *map(str, paths), "-brick", "0", "-etac",
              "-etac_blur", "0", "4", "-prefix", str(tmp_path / "x"), "-device", "cpu"])  # fmt: skip
    assert "OLS-only" in capsys.readouterr().err
