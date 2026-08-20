"""Unified axes × echoes 3-D solver: reduction to the existing paths, and dual recovery."""

import pytest
import torch

from fastfuncstuff.processing.locomoco import (
    _shift3d_axes,
    optical_flow_lk_3d,
    optical_flow_lk_3d_axes,
    optical_flow_lk_3d_multiecho,
)


def _blobby(shape=(24, 26, 22), seed=0):
    """A volume with gradient structure in every direction (no single edge orientation)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(1, *shape, generator=g)
    # smooth it into blobs so LK has a well-posed, differentiable field to track
    import torch.nn.functional as F

    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    k = k / k.sum()
    for d in range(3):
        sh = [1, 1, 1, 1, 1]
        sh[d + 2] = 5
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - d)] = 2
        pad[2 * (2 - d) + 1] = 2
        v = F.conv3d(F.pad(v.unsqueeze(1), pad, mode="replicate"), k.view(sh)).squeeze(1)
    return v


def test_one_axis_one_echo_matches_optical_flow_lk_3d():
    fixed = _blobby()
    moving = _shift3d_axes(fixed, [torch.full_like(fixed, 0.7)], [1])
    ref = optical_flow_lk_3d(fixed, moving, 1, n_levels=2, n_iters=3, window_sigma=2.0)
    got = optical_flow_lk_3d_axes(
        [fixed], [moving], [1], torch.ones(1, 1), n_levels=2, n_iters=3, window_sigma=2.0
    )[0]
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_one_axis_multi_echo_matches_multiecho_solver():
    fixed = _blobby()
    alpha = torch.tensor([1.0, 2.0, 3.0])
    fixed_list = [fixed * (1.0 - 0.2 * j) for j in range(3)]
    moving_list = [
        _shift3d_axes(f, [torch.full_like(f, 0.4 * float(alpha[j]))], [2])
        for j, f in enumerate(fixed_list)
    ]
    ref = optical_flow_lk_3d_multiecho(
        fixed_list, moving_list, alpha, 2, n_levels=2, n_iters=3, window_sigma=2.0
    )
    got = optical_flow_lk_3d_axes(
        [f.clone() for f in fixed_list],
        [m.clone() for m in moving_list],
        [2],
        alpha.view(1, 3),
        n_levels=2,
        n_iters=3,
        window_sigma=2.0,
    )[0]
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_dual_axis_recovers_two_independent_shifts():
    """Two DIFFERENT constant shifts on two axes must not leak into each other."""
    fixed = _blobby()
    d1, d2 = 0.8, -0.5
    moving = _shift3d_axes(fixed, [torch.full_like(fixed, d1), torch.full_like(fixed, d2)], [0, 1])
    u1, u2 = optical_flow_lk_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), n_levels=3, n_iters=8, window_sigma=2.0
    )
    # score the interior only: the border is clamped by padding_mode="border".
    # The solver returns the PULL displacement (moving(x+w) == fixed(x)), so building
    # `moving` by sampling `fixed` at x+d means the answer is -d.
    c = (slice(None), slice(5, -5), slice(5, -5), slice(5, -5))
    assert u1[c].mean().item() == pytest.approx(-d1, abs=0.15), u1[c].mean()
    assert u2[c].mean().item() == pytest.approx(-d2, abs=0.15), u2[c].mean()


def test_dual_axis_beats_two_separate_one_dof_solves():
    """The joint 2x2 is the point: two restricted solves absorb each other's shift."""
    fixed = _blobby(seed=3)
    d1, d2 = 0.9, -0.6
    moving = _shift3d_axes(fixed, [torch.full_like(fixed, d1), torch.full_like(fixed, d2)], [0, 1])
    c = (slice(None), slice(5, -5), slice(5, -5), slice(5, -5))
    joint = optical_flow_lk_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), n_levels=3, n_iters=8, window_sigma=2.0
    )
    sep1 = optical_flow_lk_3d(fixed, moving, 0, n_levels=3, n_iters=8, window_sigma=2.0)
    sep2 = optical_flow_lk_3d(fixed, moving, 1, n_levels=3, n_iters=8, window_sigma=2.0)
    err_joint = abs(joint[0][c].mean() + d1) + abs(joint[1][c].mean() + d2)
    err_sep = abs(sep1[c].mean() + d1) + abs(sep2[c].mean() + d2)
    assert err_joint < err_sep, f"joint {err_joint:.4f} not better than restricted {err_sep:.4f}"


def test_separability_map_flags_the_aperture_case():
    """A single straight edge cannot separate the axes; sep must collapse toward 0."""
    shape = (24, 24, 24)
    ramp = torch.arange(shape[0], dtype=torch.float32).view(1, -1, 1, 1)
    # an edge whose gradient is identical everywhere and oblique to both axes:
    # a plane varying along x+y has g0 == g1, so the 2x2 is exactly singular.
    oblique = (
        (ramp + torch.arange(shape[1], dtype=torch.float32).view(1, 1, -1, 1))
        .expand(1, *shape)
        .contiguous()
    )
    sep_edge: list[torch.Tensor] = []
    optical_flow_lk_3d_axes(
        [oblique],
        [oblique.clone()],
        [0, 1],
        torch.ones(2, 1),
        n_levels=1,
        n_iters=1,
        sep_out=sep_edge,
    )
    sep_blob: list[torch.Tensor] = []
    v = _blobby(shape)
    optical_flow_lk_3d_axes(
        [v], [v.clone()], [0, 1], torch.ones(2, 1), n_levels=1, n_iters=1, sep_out=sep_blob
    )
    c = (slice(None), slice(5, -5), slice(5, -5), slice(5, -5))
    assert sep_edge[0][c].mean() < 0.05, sep_edge[0][c].mean()
    assert sep_blob[0][c].mean() > 0.5, sep_blob[0][c].mean()


def test_axes_validation():
    v = _blobby((12, 12, 12))
    with pytest.raises(ValueError, match="must differ"):
        optical_flow_lk_3d_axes([v], [v], [1, 1], torch.ones(2, 1))
    with pytest.raises(ValueError, match="1 or 2 axes"):
        optical_flow_lk_3d_axes([v], [v], [0, 1, 2], torch.ones(3, 1))
    with pytest.raises(ValueError, match="alphas must be"):
        optical_flow_lk_3d_axes([v], [v], [0, 1], torch.ones(2, 3))


def test_xcorr_axes_one_axis_matches_multiecho_searchlight():
    from fastfuncstuff.processing.locomoco import (
        xcorr_search_flow_3d_axes,
        xcorr_search_flow_3d_multiecho,
    )

    fixed = _blobby(seed=7)
    alpha = torch.tensor([1.0, 2.0])
    fixed_list = [fixed, fixed * 0.8]
    moving_list = [
        _shift3d_axes(f, [torch.full_like(f, 0.5 * float(alpha[j]))], [1])
        for j, f in enumerate(fixed_list)
    ]
    ref, _ = xcorr_search_flow_3d_multiecho(
        fixed_list, moving_list, alpha, 1, max_shift=2.0, window_sigma=2.0
    )
    got, _ = xcorr_search_flow_3d_axes(
        fixed_list, moving_list, [1], alpha.view(1, 2), max_shift=2.0, window_sigma=2.0
    )
    assert torch.allclose(ref, got[0], atol=1e-5), (ref - got[0]).abs().max()


def test_xcorr_axes_dual_recovers_two_shifts():
    from fastfuncstuff.processing.locomoco import xcorr_search_flow_3d_axes

    fixed = _blobby(seed=11)
    d1, d2 = 1.0, -0.5
    moving = _shift3d_axes(fixed, [torch.full_like(fixed, d1), torch.full_like(fixed, d2)], [0, 1])
    w, _ = xcorr_search_flow_3d_axes(
        [fixed],
        [moving],
        [0, 1],
        torch.ones(2, 1),
        max_shift=2.0,
        window_sigma=2.0,
        trial_step=0.25,
        peak_mode="argmax",
        n_passes=3,
    )
    c = (slice(None), slice(6, -6), slice(6, -6), slice(6, -6))
    assert w[0][c].mean().item() == pytest.approx(-d1, abs=0.25), w[0][c].mean()
    assert w[1][c].mean().item() == pytest.approx(-d2, abs=0.25), w[1][c].mean()


def test_xcorr_axes_dual_multiecho_different_scaling_laws():
    """The ME dual case: axis 0 flat across echoes, axis 1 TE-scaled."""
    from fastfuncstuff.processing.locomoco import xcorr_search_flow_3d_axes

    fixed = _blobby(seed=13)
    te = torch.tensor([1.0, 2.0, 3.0])
    alphas = torch.stack([torch.ones(3), te])  # (2 axes, 3 echoes)
    g1, g2 = 0.6, -0.4
    fixed_list = [fixed * (1.0 - 0.15 * j) for j in range(3)]
    moving_list = [
        _shift3d_axes(
            f,
            [
                torch.full_like(f, float(alphas[0, j]) * g1),
                torch.full_like(f, float(alphas[1, j]) * g2),
            ],
            [0, 1],
        )
        for j, f in enumerate(fixed_list)
    ]
    w, _ = xcorr_search_flow_3d_axes(
        fixed_list,
        moving_list,
        [0, 1],
        alphas,
        max_shift=2.0,
        window_sigma=2.0,
        trial_step=0.25,
        peak_mode="argmax",
        n_passes=3,
    )
    c = (slice(None), slice(6, -6), slice(6, -6), slice(6, -6))
    assert w[0][c].mean().item() == pytest.approx(-g1, abs=0.25), w[0][c].mean()
    assert w[1][c].mean().item() == pytest.approx(-g2, abs=0.25), w[1][c].mean()


def _synth_series(shape=(26, 28, 24), nt=6, d1=None, d2=None, axes=(0, 1), seed=5):
    """A 4-D series built by shifting one volume by known per-frame amounts."""
    import numpy as np

    base = _blobby(shape, seed=seed)
    frames = []
    for t in range(nt):
        shifts = []
        for d in (d1, d2)[: len(axes)]:
            shifts.append(torch.full_like(base, float(d[t])))
        frames.append(_shift3d_axes(base, shifts, list(axes))[0])
    return np.stack([f.numpy() for f in frames], axis=-1), base


def test_run_3dacq_dual_recovers_both_axes_end_to_end():
    from fastfuncstuff.processing.locomoco import _run_3dacq_plain

    nt = 6
    d1 = [0.0, 0.6, -0.4, 0.8, -0.2, 0.5]
    d2 = [0.0, -0.3, 0.5, -0.6, 0.2, -0.4]
    data, _ = _synth_series(nt=nt, d1=d1, d2=d2, axes=(0, 1))
    res = _run_3dacq_plain(
        data,
        pe_axis=0,
        display_slice=2,
        pe_axis2=1,
        ref_mode=0,  # frame 0 is the undisplaced volume
        backend="flow",
        smooth_sigma=0,
        n_levels=3,
        n_iters=8,
        window_sigma=2.0,
        max_shift=2.0,
        trial_step=0.5,
        refine_rounds=0,
        converge=0,
        converge_rel=0,
        first_n=None,
        automask=False,
        automask_dilate=4,
        automask_sigma=3.0,
        noshift_margin=0,
        reg_sigma=0,
        peak_mode="first_peak",
        search_min_steps=5,
        save_corr_curve=None,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert res.dual and res.pe_axis == 0 and res.pe_axis2 == 1
    assert res.slice_axis == 2, "the un-encoded axis must be the display/slice axis"
    comps = res.pe_displacements()
    assert [c[0] for c in comps] == ["pe1", "pe2"]
    assert [c[1] for c in comps] == [0, 1]
    c = (slice(6, -6), slice(6, -6), slice(6, -6))
    for t in range(1, nt):
        assert comps[0][2][c][..., t].mean().item() == pytest.approx(-d1[t], abs=0.2), t
        assert comps[1][2][c][..., t].mean().item() == pytest.approx(-d2[t], abs=0.2), t


def test_run_3dacq_single_axis_unchanged_by_the_dual_generalization():
    """pe_axis2=None must reproduce the old single-axis behaviour exactly."""
    from fastfuncstuff.processing.locomoco import _run_3dacq_plain

    nt = 4
    d1 = [0.0, 0.5, -0.3, 0.7]
    data, _ = _synth_series(nt=nt, d1=d1, axes=(1,))
    kw = dict(
        display_slice=2,
        ref_mode=0,
        backend="flow",
        smooth_sigma=0,
        n_levels=2,
        n_iters=5,
        window_sigma=2.0,
        max_shift=2.0,
        trial_step=0.5,
        refine_rounds=0,
        converge=0,
        converge_rel=0,
        first_n=None,
        automask=False,
        automask_dilate=4,
        automask_sigma=3.0,
        noshift_margin=0,
        reg_sigma=0,
        peak_mode="first_peak",
        search_min_steps=5,
        save_corr_curve=None,
        device=torch.device("cpu"),
        verbose=False,
    )
    res = _run_3dacq_plain(data, pe_axis=1, **kw)
    assert not res.dual and res.pe_axis2 is None and res.sep_map is None
    comps = res.pe_displacements()
    assert len(comps) == 1 and comps[0][0] == "pe1" and comps[0][1] == 1
    assert torch.equal(comps[0][2], res.pe_displacement())
    assert res.coupling() is None
    c = (slice(6, -6), slice(6, -6), slice(6, -6))
    for t in range(1, nt):
        assert res.pe_displacement()[c][..., t].mean().item() == pytest.approx(-d1[t], abs=0.2), t


def test_coupling_report_separates_coupled_from_independent_fields():
    from fastfuncstuff.processing.locomoco import dual_field_coupling

    g = torch.Generator().manual_seed(1)
    shape = (8, 8, 8, 20)
    f1 = torch.randn(shape, generator=g)
    # perfectly coupled at a ratio of 2.5, plus a little noise
    coupled = 2.5 * f1 + 0.05 * torch.randn(shape, generator=g)
    independent = torch.randn(shape, generator=g)

    c = dual_field_coupling(f1, coupled)
    assert c["r"] > 0.99, c["r"]
    assert c["kappa"] == pytest.approx(2.5, abs=0.05), c["kappa"]
    assert c["kappa_r2"] > 0.99
    assert c["r_per_frame"].shape == (20,)
    assert c["r_per_voxel"].shape == (8, 8, 8)
    assert (c["r_per_frame"] > 0.95).all()

    i = dual_field_coupling(f1, independent)
    assert abs(i["r"]) < 0.15, i["r"]
    assert i["kappa_r2"] < 0.15, i["kappa_r2"]


def test_coupling_honours_a_mask():
    from fastfuncstuff.processing.locomoco import dual_field_coupling

    g = torch.Generator().manual_seed(2)
    f1 = torch.randn(6, 6, 6, 10, generator=g)
    f2 = f1.clone()
    mask = torch.zeros(6, 6, 6, dtype=torch.bool)
    mask[:3] = True
    # outside the mask the two fields disagree completely; the stat must not see it
    f2[3:] = torch.randn(3, 6, 6, 10, generator=g)
    c = dual_field_coupling(f1, f2, mask)
    assert c["r"] > 0.999, c["r"]
    assert c["r_per_voxel"][3:].abs().max() == 0.0, "outside-mask voxels must stay 0"


# ── CLI flag resolution ───────────────────────────────────────────────────────


def _mkseries(tmp_path, name="in.nii.gz", nt=4):
    import nibabel as nib
    import numpy as np

    data, _ = _synth_series(shape=(16, 18, 14), nt=nt, d1=[0.0] * nt, d2=[0.0] * nt, axes=(0, 1))
    p = tmp_path / name
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(p))
    return str(p)


def _run(argv):
    from fastfuncstuff.cli.locomoco import main

    return main(argv)


def test_cli_pe_dir2_implies_3d_and_writes_two_signed_flows(tmp_path):
    inp = _mkseries(tmp_path)
    stem = str(tmp_path / "o")
    rc = _run(
        [
            "-input", inp, "-prefix", stem, "-pe_dir", "x", "-pe_dir2", "y",
            "-ref", "0", "-device", "cpu", "-no_movie", "-no_warp", "-levels", "1",
            "-iters", "1",
        ]
    )  # fmt: skip
    assert rc == 0
    from pathlib import Path

    assert Path(f"{stem}_flow_pe1.nii.gz").exists()
    assert Path(f"{stem}_flow_pe2.nii.gz").exists()
    assert not Path(f"{stem}_flow.nii.gz").exists(), "two axes must not write a single _flow"
    assert Path(f"{stem}_locomoco_coupling.txt").exists()
    assert Path(f"{stem}_locomoco_sep.nii.gz").exists()


def test_cli_two_value_pe_dir_is_sugar_for_both(tmp_path):
    inp = _mkseries(tmp_path)
    stem = str(tmp_path / "o")
    rc = _run(
        [
            "-input", inp, "-prefix", stem, "-pe_dir", "x", "y", "-is_3depi",
            "-ref", "0", "-device", "cpu", "-no_movie", "-no_warp", "-levels", "1",
            "-iters", "1",
        ]
    )  # fmt: skip
    assert rc == 0
    from pathlib import Path

    assert Path(f"{stem}_flow_pe1.nii.gz").exists()
    assert Path(f"{stem}_flow_pe2.nii.gz").exists()


def test_cli_pe_dir2_alone_is_a_single_partition_axis_solve(tmp_path):
    inp = _mkseries(tmp_path)
    stem = str(tmp_path / "o")
    rc = _run(
        [
            "-input", inp, "-prefix", stem, "-pe_dir2", "y",
            "-ref", "0", "-device", "cpu", "-no_movie", "-no_warp", "-levels", "1",
            "-iters", "1",
        ]
    )  # fmt: skip
    assert rc == 0
    from pathlib import Path

    assert Path(f"{stem}_flow.nii.gz").exists(), "one axis writes the plain _flow map"
    assert not Path(f"{stem}_flow_pe2.nii.gz").exists()


def test_cli_rejects_conflicting_and_missing_directions(tmp_path):
    inp = _mkseries(tmp_path)
    stem = str(tmp_path / "o")
    base = ["-input", inp, "-prefix", stem, "-device", "cpu"]
    assert _run([*base, "-pe_dir", "x", "y", "-pe_dir2", "z"]) == 2, "both spellings at once"
    assert _run(base) == 2, "no direction at all"
    assert _run([*base, "-pe_dir", "x", "-pe_dir2", "x"]) == 2, "same axis twice"
    assert _run([*base, "-pe_dir", "x", "y", "z"]) == 2, "three directions"


def test_cli_me_bare_pe_dir_errors_with_the_rename(tmp_path, capsys):
    inp = _mkseries(tmp_path)
    stem = str(tmp_path / "o")
    rc = _run(
        [
            "-input", inp, inp, "-prefix", stem, "-pe_dir", "z", "-me_3depi",
            "-echo_times", "10", "30", "-device", "cpu",
        ]
    )  # fmt: skip
    assert rc == 2
    err = capsys.readouterr().err
    assert "-pe_dir2 z" in err, "the error must spell out the replacement"
    # ...but an explicit TE-INDEPENDENT request is a legitimate primary-PE solve
    rc2 = _run(
        [
            "-input", inp, inp, "-prefix", stem, "-pe_dir", "z", "-me_3depi",
            "-echo_times", "10", "30", "-me_flat_scaling", "-ref", "0",
            "-device", "cpu", "-no_movie", "-no_warp", "-levels", "1", "-iters", "1",
        ]
    )  # fmt: skip
    assert rc2 == 0


# ── multi-echo × two encode axes ──────────────────────────────────────────────


def _me_series(tmp_path, tes=(10.0, 20.0, 30.0), g1=None, g2=None, seed=4):
    """Echoes whose primary-PE shift is FLAT and partition shift scales with TE."""
    import nibabel as nib
    import numpy as np

    base = _blobby((26, 28, 24), seed=seed) * 1000
    nt = len(g1)
    paths = []
    for ei, te in enumerate(tes):
        a1, a2 = 1.0, te / tes[0]
        frames = [
            _shift3d_axes(
                base,
                [torch.full_like(base, a1 * g1[t]), torch.full_like(base, a2 * g2[t])],
                [0, 1],
            )[0].numpy()
            for t in range(nt)
        ]
        d = (np.stack(frames, -1) * (1.0 - 0.15 * ei)).astype(np.float32)
        p = tmp_path / f"e{ei + 1}.nii.gz"
        nib.save(nib.Nifti1Image(d, np.eye(4)), str(p))
        paths.append(str(p))
    return paths


def test_me_dual_recovers_flat_primary_and_te_scaled_partition(tmp_path):
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    tes = (10.0, 20.0, 30.0)
    g1 = [0.0, 0.5, -0.35, 0.7]
    g2 = [0.0, -0.25, 0.4, -0.3]
    paths = _me_series(tmp_path, tes, g1, g2)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]

    res = estimate_residual_flow_multiecho(
        datas,
        list(tes),
        0,  # primary PE
        2,
        pe_axis2=1,  # partition
        ref_mode=0,
        backend="flow",
        smooth_sigma=0,
        n_levels=3,
        n_iters=8,
        window_sigma=2.0,
        max_shift=2.0,
        refine_rounds=0,
        automask=False,
        learn_scaling=False,
        want_corrected=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert res.pe_axis == 1, "the scaled field is the partition axis"
    assert res.pe_axis1 == 0 and res.w_field_pe1 is not None
    c = (slice(7, -7), slice(7, -7), slice(7, -7))
    for ei, te in enumerate(tes):
        comps = res.per_echo[ei].pe_displacements()
        assert [x[1] for x in comps] == [0, 1]
        a2 = te / tes[0]
        for t in range(1, len(g1)):
            # primary PE is FLAT: every echo shifts by the same amount
            assert comps[0][2][c][..., t].mean().item() == pytest.approx(-g1[t], abs=0.15), (ei, t)
            # partition scales with TE
            assert comps[1][2][c][..., t].mean().item() == pytest.approx(-a2 * g2[t], abs=0.15), (
                ei,
                t,
            )


def test_me_dual_does_not_leak_between_axes_when_ratios_differ(tmp_path):
    """Frames where the two axes move in different ratios are the leak detector."""
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    tes = (10.0, 30.0)
    g1 = [0.0, 0.6, 0.6, -0.5]
    g2 = [0.0, -0.3, 0.5, -0.5]  # ratio to g1 changes every frame
    paths = _me_series(tmp_path, tes, g1, g2, seed=9)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    res = estimate_residual_flow_multiecho(
        datas, list(tes), 0, 2,
        pe_axis2=1, ref_mode=0, backend="flow", smooth_sigma=0, n_levels=3, n_iters=8,
        window_sigma=2.0, max_shift=2.0, refine_rounds=0, automask=False,
        learn_scaling=False, want_corrected=False, device=torch.device("cpu"), verbose=False,
    )  # fmt: skip
    c = (slice(7, -7), slice(7, -7), slice(7, -7))
    comps = res.per_echo[0].pe_displacements()
    for t in range(1, len(g1)):
        assert comps[0][2][c][..., t].mean().item() == pytest.approx(-g1[t], abs=0.15), t
        assert comps[1][2][c][..., t].mean().item() == pytest.approx(-g2[t], abs=0.15), t


def test_me_single_axis_result_shape_unchanged(tmp_path):
    """pe_axis2=None must leave the multi-echo result exactly as it was."""
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    tes = (10.0, 20.0)
    paths = _me_series(tmp_path, tes, [0.0, 0.4, -0.3], [0.0, 0.0, 0.0], seed=12)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    res = estimate_residual_flow_multiecho(
        datas, list(tes), 0, 2,
        ref_mode=0, backend="flow", smooth_sigma=0, n_levels=2, n_iters=3,
        window_sigma=2.0, max_shift=2.0, refine_rounds=0, automask=False,
        learn_scaling=False, want_corrected=False, device=torch.device("cpu"), verbose=False,
    )  # fmt: skip
    assert res.pe_axis == 0
    assert res.w_field_pe1 is None and res.pe_axis1 is None
    for r in res.per_echo:
        assert not r.dual and r.pe_axis2 is None
        assert len(r.pe_displacements()) == 1


def test_me_dual_warns_when_both_laws_are_flat(tmp_path, capsys):
    """Identical scaling laws remove the echo axis's separating power — say so."""
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    tes = (10.0, 20.0)
    paths = _me_series(tmp_path, tes, [0.0, 0.3], [0.0, 0.2], seed=15)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    estimate_residual_flow_multiecho(
        datas, list(tes), 0, 2,
        pe_axis2=1, ref_mode=0, backend="flow", smooth_sigma=0, n_levels=1, n_iters=1,
        window_sigma=2.0, max_shift=2.0, refine_rounds=0, automask=False,
        learn_scaling=False, flat_scaling=True, want_corrected=False,
        device=torch.device("cpu"), verbose=True,
    )  # fmt: skip
    assert "cannot separate them" in capsys.readouterr().out


def test_cli_me_dual_writes_per_echo_axis_maps(tmp_path):
    from pathlib import Path

    tes = (10.0, 20.0)
    paths = _me_series(tmp_path, tes, [0.0, 0.4], [0.0, -0.2], seed=17)
    stem = str(tmp_path / "m")
    rc = _run(
        [
            "-input", *paths, "-prefix", stem, "-pe_dir1", "x", "-pe_dir2", "y",
            "-me_3depi", "-echo_times", "10", "20", "-me_fixed_scaling", "-ref", "0",
            "-device", "cpu", "-no_movie", "-no_warp", "-no_corrected",
            "-levels", "1", "-iters", "1",
        ]
    )  # fmt: skip
    assert rc == 0
    for ei in (1, 2):
        assert Path(f"{stem}_e{ei}_flow_pe1.nii.gz").exists()
        assert Path(f"{stem}_e{ei}_flow_pe2.nii.gz").exists()
    assert Path(f"{stem}_locomoco_coupling.txt").exists()
    assert Path(f"{stem}_locomoco_sep.nii.gz").exists()


def test_cli_me_dual_rejects_the_modes_that_are_single_axis(tmp_path, capsys):
    tes = (10.0, 20.0)
    paths = _me_series(tmp_path, tes, [0.0, 0.4], [0.0, -0.2], seed=19)
    base = [
        "-input", *paths, "-prefix", str(tmp_path / "m"), "-pe_dir1", "x", "-pe_dir2", "y",
        "-me_3depi", "-echo_times", "10", "20", "-device", "cpu",
    ]  # fmt: skip
    assert _run([*base, "-me_interecho"]) == 2
    assert "single encode-axis only" in capsys.readouterr().err
    assert _run([*base, "-me_estimate_from", "last"]) == 2


# ── 2-D multi-slice multi-echo (no partition direction) ───────────────────────


def _slicewise_me_series(tmp_path, tes, truth, seed=6, noise=0.0, decay=0.18):
    """Echoes sharing ONE per-slice, per-frame primary-PE shift (TE-independent).

    ``truth`` is ``(nz, nt)`` voxels along x. Every echo sees the SAME shift — that is
    what makes a 2-D multi-echo run extra evidence for one displacement rather than
    several different ones.
    """
    import nibabel as nib
    import numpy as np

    nx, ny, nz = 30, 32, truth.shape[0]
    nt = truth.shape[1]
    base = _blobby((nx, ny, nz), seed=seed) * 1000
    rng = np.random.default_rng(seed + 100)
    paths = []
    for ei, _te in enumerate(tes):
        frames = []
        for t in range(nt):
            sh = torch.zeros(1, nx, ny, nz)
            for z in range(nz):
                sh[0, :, :, z] = float(truth[z, t])
            frames.append(_shift3d_axes(base, [sh], [0])[0].numpy())
        d = np.stack(frames, -1) * (1.0 - decay * ei)  # T2* decay across echoes
        if noise:
            d = d + rng.normal(0, noise, d.shape)
        p = tmp_path / f"sw_e{ei + 1}.nii.gz"
        nib.save(nib.Nifti1Image(d.astype(np.float32), np.eye(4)), str(p))
        paths.append(str(p))
    return paths


def _run_slicewise_me(paths, tes, **over):
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    kw = dict(
        ref_mode=0,
        backend="flow",
        smooth_sigma=0,
        n_levels=3,
        n_iters=8,
        window_sigma=2.0,
        max_shift=2.0,
        refine_rounds=0,
        automask=False,  # gating erodes the end slices; not what these tests measure
        learn_scaling=False,
        flat_scaling=True,
        want_corrected=False,
        slicewise=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    kw.update(over)
    return estimate_residual_flow_multiecho(datas, list(tes), 0, 2, **kw)


def test_slicewise_me_recovers_a_per_slice_field():
    import tempfile
    from pathlib import Path

    import numpy as np

    rng = np.random.default_rng(3)
    nz, nt = 10, 5
    truth = rng.uniform(-0.8, 0.8, size=(nz, nt))
    truth[:, 0] = 0.0
    tes = (10.0, 25.0, 40.0)
    with tempfile.TemporaryDirectory() as td:
        res = _run_slicewise_me(_slicewise_me_series(Path(td), tes, truth), tes)
    c = (slice(8, -8), slice(8, -8))
    for ei in range(len(tes)):
        f = res.per_echo[ei].pe_displacement()
        for z in range(nz):
            for t in range(1, nt):
                assert f[c][:, :, z, t].mean().item() == pytest.approx(-truth[z, t], abs=0.15), (
                    ei,
                    z,
                    t,
                )


def test_slicewise_me_does_not_pool_across_slices():
    """Adjacent slices moving in OPPOSITE directions must be recovered independently."""
    import tempfile
    from pathlib import Path

    import numpy as np

    nz, nt = 8, 3
    truth = np.zeros((nz, nt))
    truth[:, 1] = [0.7 if z % 2 == 0 else -0.7 for z in range(nz)]  # alternating
    truth[:, 2] = [-0.6 if z % 2 == 0 else 0.6 for z in range(nz)]
    tes = (10.0, 30.0)
    with tempfile.TemporaryDirectory() as td:
        res = _run_slicewise_me(_slicewise_me_series(Path(td), tes, truth, seed=8), tes)
    f = res.per_echo[0].pe_displacement()
    c = (slice(8, -8), slice(8, -8))
    for z in range(nz):
        for t in (1, 2):
            got = f[c][:, :, z, t].mean().item()
            assert got == pytest.approx(-truth[z, t], abs=0.2), (z, t, got)
            # a solve that pooled through-plane would average the alternating signs to ~0
            assert abs(got) > 0.3, f"slice {z} frame {t} collapsed toward zero: {got:.3f}"


def test_slicewise_me_pools_echoes_for_accuracy():
    """The multi-echo point: more echoes = more evidence for the SAME shift.

    Scored in the NOISE-dominated regime with equal-amplitude echoes, which is where the
    claim is actually testable. Pooled LK has noise variance ``sigma^2 / sum(s_e^2)``, so
    every echo with signal left lowers it. Under strong T2* decay the late echoes carry
    so little signal that the reduction is within run-to-run scatter -- real, but not
    something a test can assert (measured: 4 echoes at 18%/echo decay was a wash against
    2). Averaged over several noise draws so a single lucky seed cannot decide it.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    rng = np.random.default_rng(11)
    nz, nt = 8, 4
    truth = rng.uniform(-0.7, 0.7, size=(nz, nt))
    truth[:, 0] = 0.0
    tes = (10.0, 25.0, 40.0, 55.0)
    c = (slice(8, -8), slice(8, -8))

    def _err(res):
        f = res.per_echo[0].pe_displacement()
        return float(
            np.mean(
                [
                    abs(f[c][:, :, z, t].mean().item() + truth[z, t])
                    for z in range(nz)
                    for t in range(1, nt)
                ]
            )
        )

    many, few = [], []
    for seed in (13, 21, 29, 37, 45):
        with tempfile.TemporaryDirectory() as td:
            # decay=0: equal-amplitude echoes, so the ONLY difference is how much
            # independent evidence each solve gets.
            paths = _slicewise_me_series(Path(td), tes, truth, seed=seed, noise=150.0, decay=0.0)
            many.append(_err(_run_slicewise_me(paths, tes)))
            few.append(_err(_run_slicewise_me(paths[:2], tes[:2])))
    assert np.mean(many) < np.mean(few), (
        f"pooling 4 echoes ({np.mean(many):.4f}) should beat 2 ({np.mean(few):.4f})"
    )


def test_slicewise_rejects_a_partition_axis_and_a_pe_slice_clash():
    import tempfile
    from pathlib import Path

    import numpy as np

    tes = (10.0, 30.0)
    truth = np.zeros((6, 2))
    with tempfile.TemporaryDirectory() as td:
        paths = _slicewise_me_series(Path(td), tes, truth, seed=15)
        with pytest.raises(ValueError, match="no partition direction"):
            _run_slicewise_me(paths, tes, pe_axis2=1)
        import nibabel as nib

        from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

        datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
        with pytest.raises(ValueError, match="inside the slice plane"):
            estimate_residual_flow_multiecho(
                datas, list(tes), 2, 2,  # pe_axis == slice_axis
                slicewise=True, ref_mode=0, backend="flow", automask=False,
                learn_scaling=False, flat_scaling=True, want_corrected=False,
                device=torch.device("cpu"), verbose=False,
            )  # fmt: skip


def test_blur_and_pyramid_skip_the_slice_axis():
    from fastfuncstuff.processing.locomoco import _blur3d_b, _pyr_down3d, _pyr_min_extent

    v = torch.zeros(1, 9, 9, 9)
    v[0, 4, 4, 4] = 1.0
    full = _blur3d_b(v, 1.5)
    skipped = _blur3d_b(v, 1.5, skip_axis=2)
    # blurring all three axes spreads along z; skipping it leaves z a delta
    assert full[0, 4, 4, 3] > 1e-4
    assert skipped[0, 4, 4, 3] == 0.0, "skip_axis still blurred through plane"
    assert skipped[0, 4, 3, 4] > 0.0, "in-plane blur must still happen"

    down = _pyr_down3d(torch.zeros(1, 16, 16, 10), skip_axis=2)
    assert down.shape == (1, 8, 8, 10), "the slice axis must not be downsampled"
    # the depth guard must ignore the un-downsampled axis
    assert _pyr_min_extent((16, 16, 4), 2) == 16
    assert _pyr_min_extent((16, 16, 4), None) == 4


# ── echo count: 2 to N must work everywhere ──────────────────────────────────


def _echoes(tmp_path, n_echo, nt=3, shape=(24, 26, 10), seed=41, shift=0.4):
    """N echoes of one series with a known flat (TE-independent) per-frame shift."""
    import nibabel as nib
    import numpy as np

    base = _blobby(shape, seed=seed) * 1000
    tes = [10.0 + 15.0 * i for i in range(n_echo)]
    paths = []
    for ei in range(n_echo):
        frames = [
            _shift3d_axes(base, [torch.full_like(base, shift * t)], [0])[0].numpy()
            for t in range(nt)
        ]
        d = (np.stack(frames, -1) * (1.0 - 0.12 * ei)).astype(np.float32)
        p = tmp_path / f"n{n_echo}_e{ei}.nii.gz"
        nib.save(nib.Nifti1Image(d, np.eye(4)), str(p))
        paths.append(str(p))
    return paths, tes


@pytest.mark.parametrize("n_echo", [2, 3, 4, 5])
def test_echo_count_sweep_temporal_paths(tmp_path, n_echo):
    """Nothing may assume 3 echoes: 2..N across the 3-D, 2-D and dual solves."""
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    paths, tes = _echoes(tmp_path, n_echo)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    common = dict(
        ref_mode=0, backend="flow", smooth_sigma=0, n_levels=2, n_iters=3,
        window_sigma=2.0, max_shift=2.0, refine_rounds=0, automask=False,
        want_corrected=False, device=torch.device("cpu"), verbose=False,
    )  # fmt: skip
    variants = {
        "3d-learned": dict(learn_scaling=True),
        "3d-fixed-TE": dict(learn_scaling=False),
        "3d-flat": dict(learn_scaling=False, flat_scaling=True),
        "2d-slicewise": dict(learn_scaling=False, flat_scaling=True, slicewise=True),
        "3d-dual": dict(learn_scaling=False, pe_axis2=1),
    }
    for name, over in variants.items():
        res = estimate_residual_flow_multiecho(datas, tes, 0, 2, **common, **over)
        assert len(res.per_echo) == n_echo, name
        assert res.alpha.shape == (n_echo,), (name, res.alpha.shape)
        assert res.echo_times.shape == (n_echo,), name
        for r in res.per_echo:
            assert r.pe_displacement().shape == datas[0].shape, name
        assert np.isfinite(res.linearity_r2), name


@pytest.mark.parametrize("n_echo", [2, 3, 4, 5])
def test_echo_count_sweep_interecho_and_scaled(tmp_path, n_echo):
    """The inter-echo and estimate-from paths across 2..N echoes."""
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import (
        estimate_residual_flow_me_interecho,
        estimate_residual_flow_me_scaled,
    )

    paths, tes = _echoes(tmp_path, n_echo, seed=43)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]

    # estimate on the LAST echo (largest shifts) and scale the rest by TE ratio
    scaled = estimate_residual_flow_me_scaled(
        datas, tes, n_echo - 1, 0, 2,
        ref_mode=0, backend="flow", smooth_sigma=0, n_levels=2, n_iters=3,
        window_sigma=2.0, max_shift=2.0, refine_rounds=0, automask=False,
        device=torch.device("cpu"), verbose=False,
    )  # fmt: skip
    assert len(scaled.per_echo) == n_echo and scaled.alpha.shape == (n_echo,)

    inter = estimate_residual_flow_me_interecho(
        datas, tes, 0, 2,
        backend="xcorr", smooth_sigma=0, n_levels=2, n_iters=3, window_sigma=2.0,
        max_shift=1.5, automask=False, device=torch.device("cpu"), verbose=False,
    )  # fmt: skip
    assert len(inter.per_echo) == n_echo and inter.alpha.shape == (n_echo,)


@pytest.mark.parametrize("n_echo", [2, 3, 5])
def test_echo_count_sweep_qwarp(tmp_path, n_echo):
    """qwarp polish over 2..N echoes, single axis and dual."""
    import nibabel as nib

    from fastfuncstuff.processing.locomoco import (
        estimate_residual_flow_multiecho,
        polish_me_result,
    )

    paths, tes = _echoes(tmp_path, n_echo, shape=(22, 24, 8), seed=47)
    datas = [nib.load(p).get_fdata().astype("float32") for p in paths]
    for over in ({}, {"pe_axis2": 1}):
        res = estimate_residual_flow_multiecho(
            datas, tes, 0, 2, ref_mode=0, backend="flow", smooth_sigma=0, n_levels=1,
            n_iters=2, window_sigma=2.0, max_shift=1.5, refine_rounds=0, automask=False,
            learn_scaling=False, want_corrected=True, device=torch.device("cpu"),
            verbose=False, **over,
        )  # fmt: skip
        pol = polish_me_result(
            res, minpatch=7, n_levels=1, iters=3, slicewise=False,
            device=torch.device("cpu"), verbose=False,
        )  # fmt: skip
        assert len(pol.per_echo) == n_echo
        assert pol.alpha.shape == (n_echo,)


def test_two_echoes_linearity_r2_is_not_a_free_pass():
    """A 2-parameter model fits 2 points exactly; that must be stated, not scored."""
    from fastfuncstuff.processing.locomoco import _affine_in_te_r2

    te2 = torch.tensor([10.0, 30.0])
    assert _affine_in_te_r2(torch.randn(2, 4, 4, 4, 3), te2) == 1.0
    # with 3+ echoes it becomes a genuine check that can fail
    te3 = torch.tensor([10.0, 30.0, 50.0])
    junk = torch.randn(3, 6, 6, 6, 4)
    assert _affine_in_te_r2(junk, te3) < 0.9
