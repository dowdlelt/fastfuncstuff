"""Joint multi-echo TE-scaled PE-only qwarp polish (locomoco hand-off).

The ME 3-D-EPI partition/PE wiggle scales linearly with echo time: one shared
displacement field ``w`` and echo ``e`` sees ``alpha_e * w`` with
``alpha_e = TE_e / TE_1`` FIXED (not fitted). :func:`qwarp_pe_scaled_polish`
refines a seed ``w`` under the joint objective
``sum_e lpa(warp(source_e, alpha_e*w), base_e)``.

Correctness is judged on field recovery (``corr(field, w_true)``) -- the direct
measure -- rather than fragile absolute image-correlation margins. The alpha guard
verifies the per-echo scaling is load-bearing: with the wrong alpha a single field
cannot align a 2x-shifted echo.
"""

import pytest
import torch

from fastfuncstuff.processing.interp import warp_image
from fastfuncstuff.processing.warp import (
    QwarpConfig,
    qwarp_pe_scaled_polish,
    qwarp_pe_scaled_polish_series,
)


def _sepconv(v: torch.Tensor, k: torch.Tensor, axis: int) -> torch.Tensor:
    v = v.movedim(axis, -1)
    shape = v.shape
    flat = v.reshape(-1, 1, shape[-1])  # (N, C=1, L) for replicate pad
    pad = (k.numel() - 1) // 2
    vp = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    out = torch.zeros_like(flat)
    for i, w in enumerate(k):
        out = out + w * vp[..., i : i + shape[-1]]
    return out.reshape(shape).movedim(-1, axis)


def _smooth_volume(nz: int, ny: int, nx: int, seed: int) -> torch.Tensor:
    """Structured positive volume: short-correlation texture the cost can lock to."""
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(nz, ny, nx, generator=g)
    for axis in (0, 1, 2):
        v = _sepconv(v, torch.tensor([0.25, 0.5, 0.25]), axis)
    v = v - v.min()
    return v / v.max() + 0.05


def _smooth_pe_field(nz: int, ny: int, nx: int, amp: float) -> torch.Tensor:
    """Smooth PE (y-axis) displacement, tapered to zero at the edges (in-bounds)."""
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, 1, nz),
        torch.linspace(0, 1, ny),
        torch.linspace(0, 1, nx),
        indexing="ij",
    )
    field = amp * torch.sin(2 * torch.pi * xx) * torch.cos(torch.pi * zz)
    taper = torch.sin(torch.pi * yy) * torch.sin(torch.pi * xx).clamp(min=0)
    return (field * taper).contiguous()


def _corr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
    a = a[m]
    b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize(
    "cost,optimizer",
    [("ncc", "gn"), ("ncc", "adam"), pytest.param("lpa", "adam", marks=pytest.mark.gpu)],
)
def test_single_echo_recovers_field_and_improves_alignment(cost, optimizer):
    nz, ny, nx = 10, 24, 24
    base = _smooth_volume(nz, ny, nx, seed=0)
    w_true = _smooth_pe_field(nz, ny, nx, amp=2.0)

    # source displaced by -w_true along PE (y): warp(source, +w_true) ~= base.
    z0 = torch.zeros_like(w_true)
    source = warp_image(base, z0, -w_true, z0, mode="linear")

    cfg = QwarpConfig(minpatch=7, cost_method=cost, verb=0, optimizer=optimizer)
    warped, fields = qwarp_pe_scaled_polish(
        base[None], source[None], pe_grid_axis=1, config=cfg, n_levels=2
    )

    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[2:-2, 4:-4, 4:-4] = True
    big = interior & (w_true.abs() > 0.4)

    before = _corr(source, base, interior)
    after = _corr(warped[0], base, interior)
    assert after > before, f"alignment did not improve: {before:.3f} -> {after:.3f}"
    assert after > 0.97, f"warped should closely match base, got corr {after:.3f}"
    field = fields[0]  # (A, nz, ny, nx) -> the single encode axis
    assert _corr(field, w_true, big) > 0.85, "recovered field does not track truth"


def test_joint_scaling_aligns_both_echoes():
    """One shared field, alpha=[1,2]: both echoes align together, and polishing a
    degraded seed recovers the true field better than the seed."""
    nz, ny, nx = 10, 24, 24
    base = _smooth_volume(nz, ny, nx, seed=1)
    w_true = _smooth_pe_field(nz, ny, nx, amp=1.6)
    alpha = torch.tensor([1.0, 2.0])
    z0 = torch.zeros_like(w_true)

    base_echoes = torch.stack([base, base])
    source_echoes = torch.stack(
        [warp_image(base, z0, -float(a) * w_true, z0, mode="linear") for a in alpha]
    )

    # Seed with a degraded (70%) field, standing in for locomoco's ~90% estimate.
    seed = 0.7 * w_true

    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    warped, fields = qwarp_pe_scaled_polish(
        base_echoes,
        source_echoes,
        pe_grid_axis=1,
        alpha=alpha,
        seed_field=seed,
        config=cfg,
        n_levels=2,
    )

    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[2:-2, 4:-4, 4:-4] = True
    big = interior & (w_true.abs() > 0.3)

    for e in (0, 1):
        before = _corr(source_echoes[e], base, interior)
        after = _corr(warped[e], base, interior)
        assert after > before, f"echo {e} not improved: {before:.3f} -> {after:.3f}"
        assert after > 0.95, f"echo {e} poorly aligned: corr {after:.3f}"

    # corr is scale-invariant (the 0.7x seed already correlates 1.0 with truth), so
    # judge the recovery on magnitude: the polish must reduce the seed's under-shoot.
    def _rmse(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
        return float(((a[m] - b[m]) ** 2).mean().sqrt())

    field = fields[0]
    assert _rmse(field, w_true, big) < _rmse(seed, w_true, big), "polish did not beat seed"
    assert _corr(field, w_true, big) > 0.85


def test_wrong_alpha_leaves_long_echo_misaligned():
    """Guard: treating both echoes as alpha=1 must fail to align the 2x echo.

    A single field cannot simultaneously satisfy a 1x and a 2x shift; only the
    correct alpha lets the shared field align the long echo. Proves the per-echo
    scaling is actually used -- if the cost ignored alpha this test would pass
    spuriously (both runs identical)."""
    nz, ny, nx = 10, 24, 24
    base = _smooth_volume(nz, ny, nx, seed=2)
    w_true = _smooth_pe_field(nz, ny, nx, amp=2.0)
    z0 = torch.zeros_like(w_true)

    base_echoes = torch.stack([base, base])
    source_echoes = torch.stack(
        [
            warp_image(base, z0, -w_true, z0, mode="linear"),
            warp_image(base, z0, -2.0 * w_true, z0, mode="linear"),
        ]
    )

    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    warped_wrong, _ = qwarp_pe_scaled_polish(
        base_echoes,
        source_echoes,
        pe_grid_axis=1,
        alpha=torch.tensor([1.0, 1.0]),
        config=cfg,
        n_levels=2,
    )
    warped_right, _ = qwarp_pe_scaled_polish(
        base_echoes,
        source_echoes,
        pe_grid_axis=1,
        alpha=torch.tensor([1.0, 2.0]),
        config=cfg,
        n_levels=2,
    )

    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[2:-2, 4:-4, 4:-4] = True

    long_wrong = _corr(warped_wrong[1], base, interior)
    long_right = _corr(warped_right[1], base, interior)
    assert long_right > long_wrong + 0.03, (
        f"correct alpha should align the long echo better: "
        f"wrong={long_wrong:.3f} right={long_right:.3f}"
    )


def test_series_polishes_every_frame():
    """The 4-D series wrapper polishes each frame and returns the right shapes."""
    nz, ny, nx, T = 8, 22, 22, 3
    base = _smooth_volume(nz, ny, nx, seed=3)
    w_true = _smooth_pe_field(nz, ny, nx, amp=1.8)
    alpha = torch.tensor([1.0, 2.0])
    z0 = torch.zeros_like(w_true)

    base_echoes = torch.stack([base, base])
    # Per-frame the true field scales slightly, so each frame is a distinct problem.
    source_series = torch.empty(2, nz, ny, nx, T)
    for t in range(T):
        wt = w_true * (0.8 + 0.2 * t)
        for e, a in enumerate(alpha):
            source_series[e, ..., t] = warp_image(base, z0, -float(a) * wt, z0, mode="linear")
    seed_series = torch.zeros(nz, ny, nx, T)

    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    warped, field = qwarp_pe_scaled_polish_series(
        base_echoes,
        source_series,
        seed_series,
        pe_grid_axis=1,
        alpha=alpha,
        config=cfg,
        n_levels=2,
        show_progress=False,
    )

    assert warped.shape == (2, nz, ny, nx, T)
    assert field.shape == (1, nz, ny, nx, T)
    field = field[0]

    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[2:-2, 4:-4, 4:-4] = True
    for t in range(T):
        for e in (0, 1):
            before = _corr(source_series[e, ..., t], base, interior)
            after = _corr(warped[e, ..., t], base, interior)
            assert after >= before - 1e-3, (
                f"frame {t} echo {e} regressed: {before:.3f}->{after:.3f}"
            )


@pytest.mark.gpu
def test_series_compile_flag_matches_eager():
    """The opt-in fast path (``compile=True``) stays close to the eager path.

    On CPU ``compile=True`` is a no-op (``_maybe_compile`` returns eager, TF32 scope
    inert), so this still exercises the module-level ``_mescaled_gn_normal_eqs``
    extraction and the config plumbing -- it must be bit-identical there. On CUDA it
    runs the real compiled + TF32 path, whose GN steps are only TF32-perturbed and so
    stay close (the cost-improvement guard keeps the registration in the same place)."""
    nz, ny, nx, T = 8, 22, 22, 3
    base = _smooth_volume(nz, ny, nx, seed=5)
    w_true = _smooth_pe_field(nz, ny, nx, amp=1.6)
    alpha = torch.tensor([1.0, 1.7])
    z0 = torch.zeros_like(w_true)

    base_echoes = torch.stack([base, base])
    source_series = torch.empty(2, nz, ny, nx, T)
    for t in range(T):
        wt = w_true * (0.8 + 0.2 * t)
        for e, a in enumerate(alpha):
            source_series[e, ..., t] = warp_image(base, z0, -float(a) * wt, z0, mode="linear")
    seed_series = torch.zeros(nz, ny, nx, T)

    def _run(do_compile: bool):
        cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn", compile=do_compile)
        _, field = qwarp_pe_scaled_polish_series(
            base_echoes,
            source_series,
            seed_series,
            pe_grid_axis=1,
            alpha=alpha,
            config=cfg,
            n_levels=2,
            show_progress=False,
        )
        return field[0]

    f_eager = _run(False)
    f_fast = _run(True)
    assert torch.isfinite(f_fast).all()
    tol = 0.0  # this test runs on CPU, where compile is a no-op
    assert (f_fast - f_eager).abs().median() <= tol


# ---------------------------------------------------------------------------
# 2-D slicewise patches (2-D multi-slice acquisitions)
# ---------------------------------------------------------------------------


def _slice_discontinuous_pe_field(nz: int, ny: int, nx: int, amp: float) -> torch.Tensor:
    """In-plane-smooth PE field whose amplitude flips sign slice to slice.

    The physics of a 2-D multi-slice acquisition: every slice is sampled at its own
    instant, so its residual motion is independent of its neighbours'. A 3-D cubic
    patch spanning ``minpatch`` slices cannot represent this.
    """
    yy, xx = torch.meshgrid(torch.linspace(0, 1, ny), torch.linspace(0, 1, nx), indexing="ij")
    plane = torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy) * torch.sin(torch.pi * xx)
    per_slice = torch.tensor([1.0 if z % 2 == 0 else -1.0 for z in range(nz)])
    return (amp * per_slice[:, None, None] * plane[None]).contiguous()


def test_slicewise_beats_3d_patches_on_slice_independent_motion():
    """Single echo, per-slice-independent PE motion: 2-D patches recover it, 3-D can't.

    This is the locomoco 2-D multi-slice case. The 3-D run is not expected to fail
    outright -- it just cannot follow a field that flips sign every slice, because one
    patch covers ``minpatch`` of them.
    """
    nz, ny, nx = 12, 24, 24
    base = _smooth_volume(nz, ny, nx, seed=7)
    w_true = _slice_discontinuous_pe_field(nz, ny, nx, amp=1.2)
    z0 = torch.zeros_like(w_true)
    source = warp_image(base, z0, -w_true, z0, mode="linear")

    def _run(slicewise_axis):
        cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
        return qwarp_pe_scaled_polish(
            base[None],
            source[None],
            pe_grid_axis=1,  # PE = y
            config=cfg,
            n_levels=2,
            slicewise_axis=slicewise_axis,
        )

    warped_2d, fields_2d = _run(2)  # slicewise across z
    warped_3d, fields_3d = _run(None)
    field_2d, field_3d = fields_2d[0], fields_3d[0]  # (A, nz, ny, nx), one encode axis

    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[1:-1, 4:-4, 4:-4] = True
    big = interior & (w_true.abs() > 0.3)

    before = _corr(source, base, interior)
    c2d = _corr(warped_2d[0], base, interior)
    c3d = _corr(warped_3d[0], base, interior)
    assert c2d > before, f"slicewise did not improve alignment: {before:.3f} -> {c2d:.3f}"
    assert c2d > c3d, f"slicewise should beat 3-D patches here: {c2d:.3f} vs {c3d:.3f}"

    f2 = _corr(field_2d, w_true, big)
    f3 = _corr(field_3d, w_true, big)
    assert f2 > 0.8, f"slicewise field does not track truth: corr {f2:.3f}"
    assert f2 > f3 + 0.2, f"slicewise field should track truth far better: {f2:.3f} vs {f3:.3f}"


def test_slicewise_patches_are_one_voxel_thick():
    """The plan's geometry really is 2-D: unit extent and no through-plane basis."""
    from fastfuncstuff.processing.warp import _build_mescaled_plan

    nz, ny, nx = 10, 22, 22
    base = _smooth_volume(nz, ny, nx, seed=8)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0)
    dev = torch.device("cpu")

    plan2d = _build_mescaled_plan(base[None], 1, None, None, cfg, dev, 2, True, 2)
    plan3d = _build_mescaled_plan(base[None], 1, None, None, cfg, dev, 2, True, None)

    for level in plan2d.levels:
        assert level.nzh == 1, "slicewise patches must be one slice thick"
        assert level.nxh > 1 and level.nyh > 1
        # cubic_lite drops from 4 products to 3 once the z-modulated one is null.
        assert level.basis.shape[0] == 3
        assert (level.basis.abs().amax(dim=1) > 0).all(), "null basis row survived"
        assert level.basis.shape[1] == level.nxh * level.nyh
    for level in plan3d.levels:
        assert level.nzh == level.nxh == level.nyh
        assert level.basis.shape[0] == 4

    # Collapsing the flat axis' checkerboard parity leaves at most 4 occupied phases.
    for level in plan2d.levels:
        assert len([p for p in level.phases if p.B > 0]) <= 4


def test_slicewise_axis_must_differ_from_pe_axis():
    from fastfuncstuff.processing.warp import _build_mescaled_plan

    base = _smooth_volume(8, 20, 20, seed=9)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0)
    with pytest.raises(ValueError, match="must differ from every encode axis"):
        _build_mescaled_plan(base[None], 1, None, None, cfg, torch.device("cpu"), 2, True, 1)


def test_slicewise_single_echo_untouched_by_te_scaling():
    """E=1 must not pick up anything from the multi-echo machinery it borrows.

    The single-echo CLI wraps its result at ``E=1, alpha=[1]``; this pins that the
    TE-scaling path is inert there -- an explicit alpha of 1 is bit-identical to no
    alpha at all, and duplicating the echo (alpha=[1,1]) changes nothing but the
    batch dimension.
    """
    nz, ny, nx, T = 8, 22, 22, 2
    base = _smooth_volume(nz, ny, nx, seed=10)
    w_true = _slice_discontinuous_pe_field(nz, ny, nx, amp=1.0)
    z0 = torch.zeros_like(w_true)
    seed_series = torch.zeros(nz, ny, nx, T)

    def _run(vol, n_echo: int, alpha):
        src = torch.stack(
            [warp_image(vol, z0, -(0.9 + 0.1 * t) * w_true, z0, "linear") for t in range(T)], dim=-1
        )
        cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
        _, field = qwarp_pe_scaled_polish_series(
            vol[None].expand(n_echo, -1, -1, -1),
            src[None].expand(n_echo, -1, -1, -1, -1),
            seed_series,
            pe_grid_axis=1,
            alpha=alpha,
            config=cfg,
            n_levels=2,
            show_progress=False,
            slicewise_axis=2,
        )
        return field[0]

    f_none = _run(base, 1, None)
    f_one = _run(base, 1, torch.tensor([1.0]))
    assert torch.equal(f_none, f_one), "alpha=[1] must be identical to no alpha at E=1"

    def _q99(x):
        return float(torch.quantile(x.reshape(-1), 0.99))

    # Duplicating the echo doubles hmat, grad and the LM damping alike, so the step is
    # algebraically identical -- but only up to float32. On a volume this small, most
    # of the padded grid is air (AFNI's 9-voxel margin floor dominates the
    # ceil(0.1234*n)+1 rule here), so a good fraction of patches sit on the
    # accept/reject tie and the accepted field is chaotically sensitive to *any*
    # perturbation. Calibrate against that instead of guessing an absolute bound:
    # rescaling the data by 1+1e-6 cannot change the true displacement, so whatever it
    # moves is this problem's own noise floor. A real E-dependence would push past it
    # and would grow with the echo count; a tie-flip stays inside it.
    floor = (_run(base * (1 + 1e-6), 1, None) - f_none).abs()
    f_dup = _run(base, 2, torch.tensor([1.0, 1.0]))
    d = (f_dup - f_none).abs()
    assert float(d.median()) < 1e-5, "a duplicated echo changed the E=1 solution"
    assert _q99(d) <= max(3.0 * _q99(floor), 1e-4), (
        f"echo duplication moved the field past its own noise floor: "
        f"q99 {_q99(d):.3e} vs floor {_q99(floor):.3e}"
    )


# ── two encode axes ───────────────────────────────────────────────────────────


def _two_axis_source(nz, ny, nx, seed=21, amp_a=1.4, amp_b=-1.0):
    """A volume displaced along BOTH in-plane axes by two different smooth fields."""
    base = _smooth_volume(nz, ny, nx, seed=seed)
    wa = _smooth_pe_field(nz, ny, nx, amp=amp_a)
    wb = _smooth_pe_field(nz, ny, nx, amp=amp_b).flip(-1).contiguous()
    z = torch.zeros_like(wa)
    # channels: x <- wa, y <- wb  (warp_image takes xd, yd, zd)
    source = warp_image(base, -wa, -wb, z, mode="linear")
    return base, source, wa, wb


def test_two_axes_recovers_both_fields():
    nz, ny, nx = 10, 24, 24
    base, source, wa, wb = _two_axis_source(nz, ny, nx)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    _warped, fields = qwarp_pe_scaled_polish(
        base[None], source[None], pe_grid_axis=(0, 1), config=cfg, n_levels=3
    )
    assert fields.shape == (2, nz, ny, nx)
    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[1:-1, 4:-4, 4:-4] = True
    big_a = interior & (wa.abs() > 0.3)
    big_b = interior & (wb.abs() > 0.3)
    assert _corr(fields[0], wa, big_a) > 0.8, _corr(fields[0], wa, big_a)
    assert _corr(fields[1], wb, big_b) > 0.8, _corr(fields[1], wb, big_b)


def test_two_axes_improves_on_solving_each_axis_alone():
    """The coupled 2-axis system must beat two independent single-axis polishes."""
    nz, ny, nx = 10, 24, 24
    base, source, wa, wb = _two_axis_source(nz, ny, nx, seed=23)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[1:-1, 4:-4, 4:-4] = True

    warped_joint, _ = qwarp_pe_scaled_polish(
        base[None], source[None], pe_grid_axis=(0, 1), config=cfg, n_levels=3
    )
    warped_x, _ = qwarp_pe_scaled_polish(
        base[None], source[None], pe_grid_axis=0, config=cfg, n_levels=3
    )
    joint = _corr(warped_joint[0], base, interior)
    single = _corr(warped_x[0], base, interior)
    before = _corr(source, base, interior)
    assert joint > before
    assert joint > single, f"joint {joint:.3f} should beat one-axis-only {single:.3f}"


def test_two_axes_uses_the_coupled_gauss_newton_system():
    """GN must stay the solver, and its Hessian must be the COUPLED (A*nb)^2 one."""
    import fastfuncstuff.processing.warp as W

    nz, ny, nx = 8, 20, 20
    base, source, _, _ = _two_axis_source(nz, ny, nx, seed=25)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")

    seen: dict[str, set] = {"a": set(), "dim": set(), "calls": set()}
    orig = W._mescaled_gn_normal_eqs
    n = 0

    def spy(w, g, a4hw, bt, omega, wsum, base_hat):
        nonlocal n
        n += 1
        seen["a"].add(int(g.shape[0]))
        h, gr = orig(w, g, a4hw, bt, omega, wsum, base_hat)
        seen["dim"].add(int(h.shape[-1]))
        return h, gr

    W._mescaled_gn_normal_eqs = spy
    try:
        qwarp_pe_scaled_polish(
            base[None], source[None], pe_grid_axis=(0, 1), config=cfg, n_levels=2
        )
        assert n > 0, "the Gauss-Newton path never ran (silent fallback to adam?)"
        assert seen["a"] == {2}, seen["a"]
        dim2 = seen["dim"].copy()
        n = 0
        seen["dim"].clear()
        qwarp_pe_scaled_polish(base[None], source[None], pe_grid_axis=0, config=cfg, n_levels=2)
        dim1 = seen["dim"]
    finally:
        W._mescaled_gn_normal_eqs = orig
    # two axes => twice the parameters, one coupled system (not two separate solves)
    assert {2 * d for d in dim1} == dim2, f"{dim1} -> {dim2}"


def test_two_axes_takes_a_per_channel_alpha():
    """Each encode axis gets its OWN per-echo law: flat on one, TE-scaled on the other."""
    nz, ny, nx = 8, 20, 20
    base, source, wa, wb = _two_axis_source(nz, ny, nx, seed=27, amp_a=0.9, amp_b=-0.6)
    # echo 2 sees 2x the y-axis (partition) shift but the SAME x-axis (primary) shift
    z = torch.zeros_like(wa)
    src2 = warp_image(base, -wa, -2.0 * wb, z, mode="linear")
    alpha_ch = torch.ones(3, 2)
    alpha_ch[1] = torch.tensor([1.0, 2.0])  # y channel scales with echo
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0, optimizer="gn")
    _warped, fields = qwarp_pe_scaled_polish(
        torch.stack([base, base]),
        torch.stack([source, src2]),
        pe_grid_axis=(0, 1),
        alpha=alpha_ch,
        config=cfg,
        n_levels=3,
    )
    interior = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior[1:-1, 4:-4, 4:-4] = True
    assert _corr(fields[0], wa, interior & (wa.abs() > 0.2)) > 0.7
    assert _corr(fields[1], wb, interior & (wb.abs() > 0.2)) > 0.7


def test_per_channel_alpha_validation():
    from fastfuncstuff.processing.warp import _alpha_per_channel

    dev = torch.device("cpu")
    flat = _alpha_per_channel(torch.tensor([1.0, 2.0]), 2, dev)
    assert flat.shape == (3, 2) and torch.equal(flat[0], flat[1]), "1-D alpha broadcasts"
    per = _alpha_per_channel(torch.ones(3, 2), 2, dev)
    assert per.shape == (3, 2)
    with pytest.raises(ValueError, match="alpha must be"):
        _alpha_per_channel(torch.ones(5), 2, dev)
    with pytest.raises(ValueError, match="alpha must be"):
        _alpha_per_channel(torch.ones(2, 2), 2, dev)


def test_two_axes_rejects_a_slicewise_axis_that_is_encoded():
    base = _smooth_volume(8, 20, 20, seed=29)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0)
    with pytest.raises(ValueError, match="must differ from every encode axis"):
        qwarp_pe_scaled_polish(
            base[None], base[None], pe_grid_axis=(0, 1), config=cfg, slicewise_axis=1
        )


def test_encode_axis_validation():
    base = _smooth_volume(8, 20, 20, seed=31)
    cfg = QwarpConfig(minpatch=7, cost_method="ncc", verb=0)
    with pytest.raises(ValueError, match="1 or 2 axes"):
        qwarp_pe_scaled_polish(base[None], base[None], pe_grid_axis=(0, 1, 2), config=cfg)
    with pytest.raises(ValueError, match="distinct"):
        qwarp_pe_scaled_polish(base[None], base[None], pe_grid_axis=(1, 1), config=cfg)
