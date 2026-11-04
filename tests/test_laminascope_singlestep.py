"""Mixing voxel timeseries into column x depth cells and fitting there must give
the same curves as fitting voxels and mixing the curves -- the identity the
two-step route (ffs_deconvolve -> ffs_laminascope -iresp_prefix) relies on.

The fit is a time-axis operator and the depth mixing is a voxel-axis operator, so
they commute exactly -- but only while every unit shares one lambda. The second
test is the tripwire for that: per-voxel lambda breaks the identity, which is why
the single-step path offers a global lambda and never a per-cell one.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.builder import build_per_run_task_designs
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.smooth_basis import fit_smooth_basis, penalty_matrix
from fastfuncstuff.laminar import columns as col
from fastfuncstuff.processing.roi_depth import DepthOptions
from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

TR = 2.0
N_KNOTS = 8
K = 6


def _ribbon(tmp_path):
    """An icosphere shell written as white/pial meshes, with its voxel grid."""
    from fastfuncstuff.io.gifti import save_gifti_surface
    from tests.test_surface_voxelize import icosphere

    v, f = icosphere(4)
    r_w, r_p = 15.0, 21.0
    save_gifti_surface(tmp_path / "lh.white.surf.gii", v * r_w, f)
    save_gifti_surface(tmp_path / "lh.pial.surf.gii", v * r_p, f)
    n = 26
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ aff[:3, :3].T + aff[:3, 3]
    surfaces = resolve_ribbon_surfaces(
        None, None, None, [str(tmp_path / "lh.white.surf.gii")],
        [str(tmp_path / "lh.pial.surf.gii")], None, hemi="lh", log=None,
    )  # fmt: skip
    return surfaces, aff, (n, n, n), xyz, np.linalg.norm(xyz, axis=-1), (r_w, r_p)


def _setup(tmp_path, n_runs=2, n_tp=40, seed=0, depth_noise=0):
    """Columns, depth reads, a two-condition TENT design with per-run drift, and
    synthetic voxel timeseries on the voxels the reads touch."""
    surfaces, aff, shape3, xyz, r, (r_w, r_p) = _ribbon(tmp_path)
    resp = [np.asarray((surfaces[0].white[:, 2] > 0.0 * r_w), bool)]
    columns = col.make_columns(surfaces, resp, radius_mm=8.0, spacing_mm=6.0, seed=0)
    reads = col.vertex_depth_reads(
        surfaces, columns, aff, shape3, [K], DepthOptions(),
        mask=None, psf=False, device=torch.device("cpu"),
    )  # fmt: skip

    rng = np.random.default_rng(seed)
    onsets = [[np.array([6.0, 30.0, 54.0]) for _ in range(n_runs)],
              [np.array([18.0, 42.0, 66.0]) for _ in range(n_runs)]]  # fmt: skip
    task = build_per_run_task_designs(
        onsets, [n_tp] * n_runs, TR, basis="TENT",
        fir_window_s=[(0.0, (N_KNOTS - 1) * TR)] * 2, device=torch.device("cpu"),
    )  # fmt: skip
    n_task = task.per_run[0].shape[1]

    # Block-diagonal per-run polynomial drift ([[Block-diagonal nuisance]]).
    polys = [
        construct_polynomial_matrix(n_tp, 2, device=torch.device("cpu")) for _ in range(n_runs)
    ]
    n_nuis = polys[0].shape[1]
    total = n_tp * n_runs
    design = torch.zeros((total, n_task + n_nuis * n_runs), dtype=torch.float32)
    for i in range(n_runs):
        sl = slice(i * n_tp, (i + 1) * n_tp)
        design[sl, :n_task] = task.per_run[i]
        design[sl, n_task + i * n_nuis : n_task + (i + 1) * n_nuis] = polys[i]

    # Voxel signal: a depth-graded HRF-ish response, drift, and noise.
    depth = np.clip((r.reshape(-1)[reads.voxels] - r_w) / (r_p - r_w), 0, 1)
    t = np.arange(N_KNOTS) * TR
    hrf = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
    hrf = hrf / hrf.max()
    beta = np.concatenate([
        np.outer(0.5 + depth, hrf), np.outer((0.5 + depth) * 1.3, hrf),
    ], axis=1)  # fmt: skip
    n_v = reads.voxels.size
    coef = np.zeros((n_v, design.shape[1]), np.float64)
    coef[:, :n_task] = beta
    coef[:, n_task:] = 0.3 * rng.standard_normal((n_v, n_nuis * n_runs))
    # The noise level is a depth gradient, which is what makes a per-unit lambda a
    # depth gradient too: depth_noise=-1 puts the noise at the white end (the real
    # EPI case, deep grey darker), +1 at the pial end.
    grad = depth if depth_noise > 0 else (1.0 - depth)
    sd = (0.05 + 0.45 * grad)[:, None] if depth_noise else 0.05
    data = torch.as_tensor(
        coef @ design.numpy().T.astype(np.float64) + sd * rng.standard_normal((n_v, total)),
        dtype=torch.float32,
    )
    penalty = penalty_matrix("diff2", [N_KNOTS, N_KNOTS], [TR, TR])
    return columns, reads, design, n_task, penalty, data


def _curves_from_betas(betas: np.ndarray) -> np.ndarray:
    """(units, 2 * N_KNOTS) -> (units, N_KNOTS, 2), the trailing shape column_curves
    mixes over (time x condition), as the iresp route hands it."""
    return betas.reshape(betas.shape[0], 2, N_KNOTS).transpose(0, 2, 1)


def _fit(data, design, n_task, penalty, **kw):
    return (
        fit_smooth_basis(data, design, n_task, penalty, device=torch.device("cpu"), **kw)
        .betas.cpu()
        .numpy()
        .astype(np.float64)
    )


def test_mix_then_fit_equals_fit_then_mix_at_one_lambda(tmp_path):
    columns, reads, design, n_task, penalty, data = _setup(tmp_path)
    lam = 3.0

    # Route A, today: fit every voxel, then mix the curves into column x depth.
    vox_betas = _fit(data, design, n_task, penalty, method="fixed", lam=lam)
    route_a = col.column_curves(reads, columns, _curves_from_betas(vox_betas), K).values

    # Route B, single step: mix the timeseries into cells, then fit each cell.
    cells = col.column_curves(reads, columns, data.numpy().astype(np.float64), K)
    n_c = cells.values.shape[0]
    flat = cells.values.reshape(n_c * K, -1)
    ok = np.isfinite(flat).all(1)
    cell_betas = np.full((n_c * K, n_task), np.nan)
    cell_betas[ok] = _fit(
        torch.as_tensor(flat[ok], dtype=torch.float32), design, n_task, penalty,
        method="fixed", lam=lam,
    )  # fmt: skip
    route_b = _curves_from_betas(cell_betas).reshape(n_c, K, N_KNOTS, 2)

    good = np.isfinite(route_a) & np.isfinite(route_b)
    assert good.sum() > 100, "the synthetic ribbon gave almost no complete cells"
    # One ULP on a compiled/chunked path, as the no-bit-equality rule asks.
    np.testing.assert_allclose(route_b[good], route_a[good], rtol=1e-5, atol=1e-5)


def _depth_profile(curves: np.ndarray) -> np.ndarray:
    """Peak of the baseline condition per depth, averaged over columns."""
    peak = np.nanmax(curves[..., 0], axis=2)  # (C, K)
    return np.nanmean(peak, axis=0)


def test_per_voxel_lambda_lets_the_noise_gradient_set_the_depth_profile(tmp_path):
    """Why the single-step path takes ONE lambda for every cell.

    The signal is identical in both halves of this test; only which END of the
    ribbon is noisy flips. One lambda is a single linear operator, so it returns
    the same (depth-blurred) profile either way. A per-unit lambda shrinks the
    noisy end harder, so the measured pial/deep ratio moves in OPPOSITE directions
    with the noise gradient -- the profile is reporting the noise, not the
    physiology. Note the blurred ratio is the honest target: the depth PSF, not the
    smoother, is what accounts for mixing across depth bins.
    """
    ratios = {}
    for sign in (-1, 1):
        columns, reads, design, n_task, penalty, data = _setup(tmp_path, depth_noise=sign, seed=1)

        def route_a(dat=data, des=design, nt=n_task, pen=penalty, cols=columns, rd=reads, **kw):
            betas = _fit(dat, des, nt, pen, **kw)
            return col.column_curves(rd, cols, _curves_from_betas(betas), K).values

        a_fixed = route_a(method="fixed", lam=3.0)
        a_reml = route_a(method="reml")

        # One lambda still commutes with the mixing, noise gradient or not.
        cells = col.column_curves(reads, columns, data.numpy().astype(np.float64), K)
        n_c = cells.values.shape[0]
        flat = cells.values.reshape(n_c * K, -1)
        ok = np.isfinite(flat).all(1)
        out = np.full((n_c * K, n_task), np.nan)
        out[ok] = _fit(torch.as_tensor(flat[ok], dtype=torch.float32), design, n_task,
                       penalty, method="fixed", lam=3.0)  # fmt: skip
        b_fixed = _curves_from_betas(out).reshape(n_c, K, N_KNOTS, 2)
        good = np.isfinite(a_fixed) & np.isfinite(b_fixed)
        assert good.sum() > 100
        assert np.abs(b_fixed[good] - a_fixed[good]).max() < 1e-5 * max(
            np.abs(a_fixed[good]).max(), 1.0
        )

        p_fixed, p_reml = _depth_profile(a_fixed), _depth_profile(a_reml)
        assert np.all(np.isfinite(p_fixed)) and np.all(np.isfinite(p_reml))
        ratios[sign] = (p_fixed[-1] / p_fixed[0], p_reml[-1] / p_reml[0])

    (fix_lo, reml_lo), (fix_hi, reml_hi) = ratios[-1], ratios[1]
    # One lambda: the same signal gives the same profile, whichever end is noisy.
    assert fix_lo == pytest.approx(fix_hi, rel=0.05), (
        f"one lambda should not care where the noise is: {fix_lo:.3f} vs {fix_hi:.3f}"
    )
    # Per-voxel lambda: the two noise gradients straddle it, noisy-deep above and
    # noisy-pial below, and the spread dwarfs one lambda's. The move is not
    # symmetric -- shrinking the small deep amplitude has more leverage on the
    # ratio than shrinking the large pial one (measured 3.05 / 2.33 vs 2.42).
    assert reml_lo > fix_lo and reml_hi < fix_hi, (
        f"per-voxel lambda should track the noise gradient; pial/deep came out "
        f"noisy-deep {reml_lo:.3f}, noisy-pial {reml_hi:.3f}, one-lambda ~{fix_lo:.3f}"
    )
    spread_fixed = abs(fix_lo - fix_hi) / fix_lo
    spread_reml = abs(reml_lo - reml_hi) / reml_lo
    assert spread_reml > 10 * spread_fixed and spread_reml > 0.15, (
        f"the noise gradient moved the per-voxel-lambda profile by {spread_reml:.1%} "
        f"and the one-lambda profile by {spread_fixed:.1%}"
    )


def _write_runs(tmp_path, n_runs=2, n_tp=60, seed=3):
    """Synthetic ribbon runs with two conditions, plus their BIDS events."""
    import nibabel as nib

    surfaces, aff, shape3, xyz, r, (r_w, r_p) = _ribbon(tmp_path)
    n = shape3[0]
    depth = np.clip((r - r_w) / (r_p - r_w), 0, 1)
    in_ribbon = (r > r_w - 2.0) & (r < r_p + 2.0)
    t = np.arange(N_KNOTS) * TR
    hrf = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
    hrf = hrf / hrf.max()
    onsets = {"base": np.array([8.0, 44.0, 80.0]), "test": np.array([26.0, 62.0, 98.0])}
    gain = {"base": 1.0, "test": 1.4}
    rng = np.random.default_rng(seed)
    files, rows = [], []
    for run in range(n_runs):
        series = np.zeros((*shape3, n_tp), np.float64)
        for lab, ons in onsets.items():
            amp = (0.5 + depth) * gain[lab] * in_ribbon
            for o in ons:
                for j, h in enumerate(hrf):
                    tp = int(round((o + j * TR) / TR))
                    if 0 <= tp < n_tp:
                        series[..., tp] += amp * h
            for o in ons:
                rows.append({"run": run, "onset": float(o), "duration": 1.0, "trial_type": lab})
        # A mean to scale against, slow drift, and noise.
        series = 1000.0 * (1.0 + 0.01 * series)
        series += 5.0 * np.linspace(-1, 1, n_tp) * in_ribbon[..., None]
        series += 1.0 * rng.standard_normal(series.shape)
        path = tmp_path / f"run{run + 1}.nii.gz"
        img = nib.Nifti1Image(series.astype(np.float32), aff)
        img.header.set_zooms((2.0, 2.0, 2.0, TR))
        nib.save(img, path)
        files.append(str(path))
        ev = tmp_path / f"run{run + 1}_events.tsv"
        mine = [x for x in rows if x["run"] == run]
        ev.write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{x['onset']}\t{x['duration']}\t{x['trial_type']}\n" for x in mine)
        )
    resp = (in_ribbon & (xyz[..., 2] > 0.0)).astype(np.float32)
    nib.save(nib.Nifti1Image(resp, aff), tmp_path / "resp.nii.gz")
    return files, [str(tmp_path / f"run{i + 1}_events.tsv") for i in range(n_runs)], n


_MESH = ("-white_mesh", "lh.white.surf.gii", "-pial_mesh", "lh.pial.surf.gii")


def _common(tmp_path, verb="0"):
    return [
        "-white_mesh", str(tmp_path / "lh.white.surf.gii"),
        "-pial_mesh", str(tmp_path / "lh.pial.surf.gii"),
        "-responsive", str(tmp_path / "resp.nii.gz"),
        "-column_radius", "8", "-column_spacing", "6", "-depths", "6",
        "-compare", "ab", "base", "test",
        "-models", "null", "superficial", "-max_iter", "3", "-rescue", "0",
        "-max_columns", "4", "-seed", "0", "-device", "cpu", "-verb", verb,
    ]  # fmt: skip


@pytest.mark.slow  # 150s: two full CLI runs of deconvolve + laminascope
def test_single_step_matches_deconvolve_then_laminascope(tmp_path):
    """The whole point of the single-step path: at one fixed lambda it must give the
    same column x depth curves as fitting voxels with ffs_deconvolve and averaging
    them, because the fit and the depth mixing commute. This is the end-to-end form
    of test_mix_then_fit_equals_fit_then_mix_at_one_lambda, through both CLIs."""
    from fastfuncstuff.cli import deconvolve, laminascope

    runs, events, _ = _write_runs(tmp_path)
    lam = "3.0"
    window = ["0", str((N_KNOTS - 1) * TR)]

    # Route A: voxel curves from ffs_deconvolve at a fixed lambda, then the old path.
    assert deconvolve.main([
        "-input", *runs, "-events", *events, "-model", "TENT",
        "-window", f"0,{(N_KNOTS - 1) * TR}", "-tent-smooth", lam, "-polort", "1",
        "-do_scale", "-prefix", str(tmp_path / "dec"), "-verb", "0", "-device", "cpu",
    ]) == 0  # fmt: skip
    a = str(tmp_path / "A")
    assert laminascope.main([
        "-iresp_prefix", str(tmp_path / "dec"), "-window", *window, "-duration", "1",
        *_common(tmp_path), "-prefix", a,
    ]) == 0  # fmt: skip

    # Route B: one step, same lambda and the same window.
    b = str(tmp_path / "B")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-window", *window,
        "-tent_lambda", lam, "-polort", "1", *_common(tmp_path), "-prefix", b,
    ]) == 0  # fmt: skip

    za, zb = np.load(f"{a}_columns.npz"), np.load(f"{b}_columns.npz")
    assert np.array_equal(za["fitted"], zb["fitted"]), "the two routes fitted different columns"
    for key in ("ab_baseline_K6", "ab_test_K6"):
        ca, cb = za[key], zb[key]
        assert ca.shape == cb.shape and np.isfinite(ca).all()
        # Measured 2.4-2.8e-6 relative (max |diff| 4e-6 on a 1.7 peak): the floor is
        # ffs_deconvolve writing its iresp files as float32, which route B never
        # round-trips through. Not bit-equality -- both routes chunk.
        np.testing.assert_allclose(cb, ca, rtol=3e-5, atol=3e-5)


@pytest.mark.slow  # 40s: one CLI run
def test_single_step_chooses_its_own_lambda_and_reports_the_cells(tmp_path, capsys):
    from fastfuncstuff.cli import laminascope

    runs, events, _ = _write_runs(tmp_path)
    out = str(tmp_path / "C")
    argv = [
        "-input", *runs, "-events", *events,
        "-window", "0", str((N_KNOTS - 1) * TR), "-polort", "1",
        *_common(tmp_path, verb="1"), "-prefix", out,
    ]  # fmt: skip
    assert laminascope.main(argv) == 0
    log = capsys.readouterr().out
    assert "global lambda log10" in log and "column x depth cell" in log
    import json

    meta = json.load(open(f"{out}.json"))
    assert any("global lambda" in n for n in meta["notes"])
    # The duration came from the events, not from a flag the user had to repeat.
    assert meta["argv"] and "-duration" not in argv


@pytest.mark.slow  # 45s: one CLI run
def test_differing_event_durations_drive_the_two_events_separately(tmp_path):
    """Phase 3: the compared conditions' events may differ in length. Each side of a
    comparison drives its own event, taken from the events themselves."""
    import json

    from fastfuncstuff.cli import laminascope

    runs, events, _ = _write_runs(tmp_path)
    for ev in events:  # 'test' events are twice as long as 'base'
        lines = open(ev).read().splitlines()
        out = [lines[0]]
        for ln in lines[1:]:
            onset, _dur, lab = ln.split("\t")
            out.append(f"{onset}\t{2.0 if lab == 'test' else 1.0}\t{lab}")
        open(ev, "w").write("\n".join(out) + "\n")

    out = str(tmp_path / "D")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-window", "0", str((N_KNOTS - 1) * TR),
        "-tent_lambda", "3.0", "-polort", "1", *_common(tmp_path), "-prefix", out,
    ]) == 0  # fmt: skip
    meta = json.load(open(f"{out}.json"))
    assert meta["durations"]["ab"] == [1.0, 2.0], "the events' own lengths should reach the model"

    # An explicit -duration still wins, and one value still means both.
    out2 = str(tmp_path / "E")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-window", "0", str((N_KNOTS - 1) * TR),
        "-duration", "1.5", "-tent_lambda", "3.0", "-polort", "1",
        *_common(tmp_path), "-prefix", out2,
    ]) == 0  # fmt: skip
    # Recorded as the pair actually used, so the provenance says what drove each event.
    assert json.load(open(f"{out2}.json"))["durations"]["ab"] == [1.5, 1.5]


def test_averaging_conditions_of_different_lengths_is_refused(tmp_path):
    """A comparison side may average several labels, but averaging events of
    different lengths gives a curve that is not a response to any one event."""
    from fastfuncstuff.cli import laminascope

    runs, events, _ = _write_runs(tmp_path)
    for ev in events:  # split 'base' into two labels of different lengths
        lines = open(ev).read().splitlines()
        out, n = [lines[0]], 0
        for ln in lines[1:]:
            onset, _dur, lab = ln.split("\t")
            if lab == "base":
                n += 1
                out.append(f"{onset}\t{1.0 if n % 2 else 3.0}\tbase{1 if n % 2 else 2}")
            else:
                out.append(f"{onset}\t1.0\t{lab}")
        open(ev, "w").write("\n".join(out) + "\n")

    rc = laminascope.main([
        "-input", *runs, "-events", *events, "-window", "0", str((N_KNOTS - 1) * TR),
        "-tent_lambda", "3.0", "-polort", "1",
        *[a if a != "base" else "base1,base2" for a in _common(tmp_path)],
        "-prefix", str(tmp_path / "F"),
    ])  # fmt: skip
    assert rc == 1


def _write_three_condition_runs(tmp_path, n_runs=2, n_tp=70, seed=7):
    """Three conditions, the third interleaved close enough to the other two that
    leaving it out of the design contaminates their curves."""
    import nibabel as nib

    _surf, aff, shape3, xyz, r, (r_w, r_p) = _ribbon(tmp_path)
    depth = np.clip((r - r_w) / (r_p - r_w), 0, 1)
    in_ribbon = (r > r_w - 2.0) & (r < r_p + 2.0)
    t = np.arange(N_KNOTS) * TR
    hrf = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
    hrf = hrf / hrf.max()
    onsets = {
        "base": np.array([6.0, 48.0, 90.0]),
        "test": np.array([20.0, 62.0, 104.0]),
        # Close behind every other event, so its response overlaps theirs.
        "other": np.array([12.0, 26.0, 54.0, 68.0, 96.0, 110.0]),
    }
    gain = {"base": 1.0, "test": 1.4, "other": 2.0}
    rng = np.random.default_rng(seed)
    files, evs = [], []
    for run in range(n_runs):
        series = np.zeros((*shape3, n_tp), np.float64)
        rows = []
        for lab, ons in onsets.items():
            amp = (0.5 + depth) * gain[lab] * in_ribbon
            for o in ons:
                for j, h in enumerate(hrf):
                    tp = int(round((o + j * TR) / TR))
                    if 0 <= tp < n_tp:
                        series[..., tp] += amp * h
                rows.append((o, lab))
        series = 1000.0 * (1.0 + 0.01 * series)
        series += 0.5 * rng.standard_normal(series.shape)
        img = nib.Nifti1Image(series.astype(np.float32), aff)
        img.header.set_zooms((2.0, 2.0, 2.0, TR))
        path = tmp_path / f"t{run + 1}.nii.gz"
        nib.save(img, path)
        files.append(str(path))
        ev = tmp_path / f"t{run + 1}_events.tsv"
        ev.write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{o}\t1.0\t{lab}\n" for o, lab in sorted(rows))
        )
        evs.append(str(ev))
    resp = (in_ribbon & (xyz[..., 2] > 0.0)).astype(np.float32)
    nib.save(nib.Nifti1Image(resp, aff), tmp_path / "resp.nii.gz")
    return files, evs


@pytest.mark.slow  # 60s: two CLI runs
def test_conditions_outside_the_comparison_are_still_modelled(tmp_path):
    """Every condition in the events file gets its own TENT columns, whether or not a
    comparison names it. It has to: its response overlaps the compared ones, so
    leaving it out of the design pushes its variance into their curves. The laminar
    stage then takes only the two the comparison asks for -- modelled, not of
    interest."""
    import json

    from fastfuncstuff.cli import laminascope

    runs, events = _write_three_condition_runs(tmp_path)
    shared = [
        "-input", *runs, "-events", *events, "-window", "0", str((N_KNOTS - 1) * TR),
        "-tent_lambda", "0.5", "-polort", "1", *_common(tmp_path),
    ]  # fmt: skip

    full = str(tmp_path / "G")
    assert laminascope.main([*shared, "-prefix", full]) == 0
    # Three conditions x knots in the task block, though only two are compared.
    assert json.load(open(f"{full}.json"))["durations"] == {"ab": [1.0, 1.0]}

    # The same fit with the third condition dropped from the design entirely.
    ignored = str(tmp_path / "H")
    assert laminascope.main([*shared, "-event_ignore", "other", "-prefix", ignored]) == 0

    zf, zi = np.load(f"{full}_columns.npz"), np.load(f"{ignored}_columns.npz")
    assert np.array_equal(zf["fitted"], zi["fitted"])
    # The compared conditions' curves move when 'other' is left out: that difference
    # IS the contamination, so modelling it is not optional. Measured 55-60% of peak
    # on this fixture (overlapping events at a 2 s TR).
    base_f, base_i = zf["ab_baseline_K6"], zi["ab_baseline_K6"]
    rel = np.abs(base_i - base_f).max() / np.abs(base_f).max()
    assert rel > 0.05, (
        f"dropping an overlapping condition changed the compared curves by only "
        f"{rel:.1%}; the fixture is not exercising the contamination it claims to"
    )


def _write_mixed_duration_runs(tmp_path, n_runs=2, n_tp=90, seed=11):
    """Two conditions whose stimuli are 3 s and 10 s long, so their responses need
    windows of different lengths."""
    import nibabel as nib

    _surf, aff, shape3, xyz, r, (r_w, r_p) = _ribbon(tmp_path)
    depth = np.clip((r - r_w) / (r_p - r_w), 0, 1)
    in_ribbon = (r > r_w - 2.0) & (r < r_p + 2.0)
    spec = {"short": (3.0, np.array([8.0, 70.0, 132.0]), 1.0),
            "long": (10.0, np.array([32.0, 94.0, 156.0]), 1.3)}  # fmt: skip
    rng = np.random.default_rng(seed)
    files, evs = [], []
    for run in range(n_runs):
        series = np.zeros((*shape3, n_tp), np.float64)
        rows = []
        for lab, (dur, onsets, gain) in spec.items():
            # A boxcar of the stimulus length convolved with a canonical-ish HRF, so a
            # 10 s stimulus genuinely needs a longer window than a 3 s one.
            t = np.arange(0.0, dur + 26.0, TR)
            h = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
            box = np.ones(max(1, int(round(dur / TR))))
            kern = np.convolve(h / h.max(), box)[: t.size]
            kern = kern / kern.max()
            amp = (0.5 + depth) * gain * in_ribbon
            for o in onsets:
                for j, v in enumerate(kern):
                    tp = int(round(o / TR)) + j
                    if 0 <= tp < n_tp:
                        series[..., tp] += amp * v
                rows.append((o, dur, lab))
        series = 1000.0 * (1.0 + 0.01 * series)
        series += 0.5 * rng.standard_normal(series.shape)
        img = nib.Nifti1Image(series.astype(np.float32), aff)
        img.header.set_zooms((2.0, 2.0, 2.0, TR))
        path = tmp_path / f"m{run + 1}.nii.gz"
        nib.save(img, path)
        files.append(str(path))
        ev = tmp_path / f"m{run + 1}_events.tsv"
        ev.write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{o}\t{d}\t{lab}\n" for o, d, lab in sorted(rows))
        )
        evs.append(str(ev))
    resp = (in_ribbon & (xyz[..., 2] > 0.0)).astype(np.float32)
    nib.save(nib.Nifti1Image(resp, aff), tmp_path / "resp.nii.gz")
    return files, evs


def _mixed_common(tmp_path, verb="0"):
    return [
        "-white_mesh", str(tmp_path / "lh.white.surf.gii"),
        "-pial_mesh", str(tmp_path / "lh.pial.surf.gii"),
        "-responsive", str(tmp_path / "resp.nii.gz"),
        "-column_radius", "8", "-column_spacing", "6", "-depths", "6",
        "-compare", "sl", "short", "long",
        "-models", "null", "superficial", "-max_iter", "3", "-rescue", "0",
        "-max_columns", "4", "-seed", "0", "-device", "cpu", "-verb", verb,
    ]  # fmt: skip


@pytest.mark.slow  # 50s: two CLI runs
def test_conditions_of_different_length_get_their_own_windows(tmp_path):
    """A 3 s and a 10 s stimulus in one comparison. Each condition's window is
    estimated from its own event duration, the two response blocks are fitted and
    concatenated at their own lengths, and the model drives each event with its own
    duration."""
    import json

    from fastfuncstuff.cli import laminascope

    runs, events = _write_mixed_duration_runs(tmp_path)
    auto = str(tmp_path / "W")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-tent_lambda", "1.0", "-polort", "1",
        *_mixed_common(tmp_path), "-prefix", auto,
    ]) == 0  # fmt: skip

    meta = json.load(open(f"{auto}.json"))
    w_short, w_long = meta["windows"]["short"], meta["windows"]["long"]
    assert w_long[1] > w_short[1], f"the 10 s stimulus needs the longer window: {meta['windows']}"
    assert meta["durations"]["sl"] == [3.0, 10.0]

    z = np.load(f"{auto}_columns.npz")
    n_short, n_long = z["times_short"].size, z["times_long"].size
    assert n_long > n_short
    # The curves keep their own lengths all the way through.
    assert z["sl_baseline_K6"].shape[-1] == n_short
    assert z["sl_test_K6"].shape[-1] == n_long
    # A single shared grid is no longer meaningful, so the legacy key is absent.
    assert "times" not in z.files

    # Explicit per-condition windows, deconvolve's spelling. They are POSITIONAL
    # against the condition order, which is trial_type's (alphabetical: long, short)
    # unless -labels sets it -- so the first pair here is 'long'.
    explicit = str(tmp_path / "X")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-window", "0,28", "0,14",
        "-tent_lambda", "1.0", "-polort", "1", *_mixed_common(tmp_path), "-prefix", explicit,
    ]) == 0  # fmt: skip
    ze = np.load(f"{explicit}_columns.npz")
    assert list(ze["labels"]) == ["long", "short"]
    assert ze["times_long"].size == 15 and ze["times_short"].size == 8


def test_a_window_count_that_does_not_match_the_conditions_is_refused(tmp_path):
    from fastfuncstuff.cli import laminascope

    runs, events = _write_mixed_duration_runs(tmp_path)
    rc = laminascope.main([
        "-input", *runs, "-events", *events, "-window", "0,14", "0,28", "0,30",
        "-tent_lambda", "1.0", "-polort", "1", *_mixed_common(tmp_path),
        "-prefix", str(tmp_path / "Y"),
    ])  # fmt: skip
    assert rc == 1


def test_a_window_starting_after_the_onset_is_refused_at_the_cli(tmp_path):
    from fastfuncstuff.cli import laminascope

    runs, events = _write_mixed_duration_runs(tmp_path)
    rc = laminascope.main([
        "-input", *runs, "-events", *events, "-window", "4,18",
        "-tent_lambda", "1.0", "-polort", "1", *_mixed_common(tmp_path),
        "-prefix", str(tmp_path / "Z"),
    ])  # fmt: skip
    assert rc == 1


@pytest.mark.slow  # 90s: deconvolve + two laminascope runs
def test_per_condition_windows_agree_between_the_two_routes(tmp_path):
    """The equivalence gate, extended to per-condition windows.

    With per-condition windows ffs_deconvolve writes each condition at its OWN length
    (long 15 samples, short 8 here), so the iresp files legitimately differ in length;
    requiring one length is what used to refuse them outright. Both routes must still
    agree at one fixed lambda."""
    from fastfuncstuff.cli import deconvolve, laminascope

    runs, events = _write_mixed_duration_runs(tmp_path)
    lam = "1.0"
    # trial_type order is alphabetical: long, then short.
    win_dec = ["0,28", "0,14"]

    assert deconvolve.main([
        "-input", *runs, "-events", *events, "-model", "TENT", "-window", *win_dec,
        "-tent-smooth", lam, "-polort", "1", "-do_scale",
        "-prefix", str(tmp_path / "mdec"), "-verb", "0", "-device", "cpu",
    ]) == 0  # fmt: skip

    a = str(tmp_path / "MA")
    assert laminascope.main([
        "-iresp_prefix", str(tmp_path / "mdec"), "-window", *win_dec, "-duration", "3", "10",
        *_mixed_common(tmp_path), "-prefix", a,
    ]) == 0  # fmt: skip
    b = str(tmp_path / "MB")
    assert laminascope.main([
        "-input", *runs, "-events", *events, "-window", *win_dec,
        "-tent_lambda", lam, "-polort", "1", *_mixed_common(tmp_path), "-prefix", b,
    ]) == 0  # fmt: skip

    za, zb = np.load(f"{a}_columns.npz"), np.load(f"{b}_columns.npz")
    assert np.array_equal(za["fitted"], zb["fitted"])
    assert za["times_short"].size == 8 and za["times_long"].size == 15
    for key in ("sl_baseline_K6", "sl_test_K6"):
        ca, cb = za[key], zb[key]
        assert ca.shape == cb.shape and np.isfinite(ca).all()
        np.testing.assert_allclose(cb, ca, rtol=3e-4, atol=3e-4)

    # The short condition's curve runs to its own window's end, not to a flat tail.
    short = za["sl_baseline_K6"]
    assert np.abs(short[..., -1]).max() > 1e-3 * np.abs(short).max()
