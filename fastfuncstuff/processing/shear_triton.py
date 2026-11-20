"""Fused Triton kernel for the 1D fractional shears of rigid resampling.

A direct GPU analogue of AFNI's ``fr_gpu_rota.cu:fr_shear_kernel``: one thread
per output voxel reads a strided row, applies the AFNI interpolation weights
inline, and writes one value. This replaces the multi-temporary ``torch.gather``
path (which is memory-bound at ~1.6 GB/s on this access pattern) and runs ~60x
faster, bit-for-bit identical.

The driver batches over volumes: the 4-shear plan is decomposed once for all
volumes (``rigid_matrix_to_shears``), then each of the 4 shear *steps* is a
single kernel launch over every volume's voxels (per-volume axis/coeffs read
from device arrays). CUDA only; callers fall back to the gather path elsewhere.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

from .shear import fft_shear_rows_batched, rigid_matrix_to_shears, split_fourier_mode

# Triton hashes its whole installation on the first kernel launch of every
# process (~1s).  Do this before any @triton.jit function can be launched.
install_triton_key_cache()

# interp mode ids (match AFNI MRI_* where they exist)
_MODE_ID = {"linear": 1, "cubic": 2, "quintic": 4, "heptic": 5, "wsinc5": 72}


@triton.jit
def _shear_step_kernel(
    inp,
    outp,
    ax_ptr,
    a_ptr,
    b_ptr,
    s_ptr,
    B,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    blocks_per_vol,
    step,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    vol = pid // blocks_per_vol
    blk = pid % blocks_per_vol
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    nxyz = nx * ny * nz
    m = (idx < nxyz) & (vol < B)

    ii = idx % nx
    jj = (idx // nx) % ny
    kk = idx // (nx * ny)
    nx2 = 0.5 * (nx - 1)
    ny2 = 0.5 * (ny - 1)
    nz2 = 0.5 * (nz - 1)

    ax = tl.load(ax_ptr + vol * 4 + step)
    a = tl.load(a_ptr + vol * 4 + step)
    b = tl.load(b_ptr + vol * 4 + step)
    s = tl.load(s_ptr + vol * 4 + step)

    # axis-dependent geometry (runtime select; one volume per block)
    af = tl.where(
        ax == 0,
        a * (jj - ny2) + b * (kk - nz2) + s,
        tl.where(ax == 1, a * (ii - nx2) + b * (kk - nz2) + s, a * (ii - nx2) + b * (jj - ny2) + s),
    )
    n = tl.where(ax == 0, nx, tl.where(ax == 1, ny, nz))
    pos = tl.where(ax == 0, ii, tl.where(ax == 1, jj, kk))
    stride = tl.where(ax == 0, 1, tl.where(ax == 1, nx, nx * ny))

    voff = vol * nxyz
    base = voff + idx - pos * stride
    af = -af
    ia = tl.floor(af).to(tl.int32)
    x = af - ia
    ix = pos + ia

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    x2 = x * x
    x2m1 = x2 - 1.0
    x2m4 = x2 - 4.0
    x2m9 = x2 - 9.0
    PIF = 3.14159265358979

    # Fixed loop over the widest support (wsinc5); weights are 0 outside each
    # mode's taps. MODE and k are constexpr, so the compiler prunes to the
    # mode's actual tap count.
    for k in tl.static_range(-4, 6):
        t = ix + k
        inb = (t >= 0) & (t < n) & m
        val = tl.load(inp + base + t * stride, mask=inb, other=0.0)

        w = 0.0
        if MODE == 1:  # linear, taps 0..1
            if k == 0:
                w = 1.0 - x
            elif k == 1:
                w = x
        elif MODE == 2:  # cubic, taps -1..2
            if k == -1:
                w = -(x) * (x - 1.0) * (x - 2.0) * 0.16666666666
            elif k == 0:
                w = 3.0 * (x + 1.0) * (x - 1.0) * (x - 2.0) * 0.16666666666
            elif k == 1:
                w = -3.0 * x * (x + 1.0) * (x - 2.0) * 0.16666666666
            elif k == 2:
                w = x * (x + 1.0) * (x - 1.0) * 0.16666666666
        elif MODE == 4:  # quintic, taps -2..3
            if k == -2:
                w = x * x2m1 * (2.0 - x) * (x - 3.0) * 0.008333333
            elif k == -1:
                w = x * x2m4 * (x - 1.0) * (x - 3.0) * 0.041666667
            elif k == 0:
                w = x2m4 * x2m1 * (3.0 - x) * 0.083333333
            elif k == 1:
                w = x * x2m4 * (x + 1.0) * (x - 3.0) * 0.083333333
            elif k == 2:
                w = x * x2m1 * (x + 2.0) * (3.0 - x) * 0.041666667
            elif k == 3:
                w = x * x2m1 * x2m4 * 0.008333333
        elif MODE == 5:  # heptic, taps -3..4
            if k == -3:
                w = x * x2m1 * x2m4 * (x - 3.0) * (4.0 - x) * 0.0001984126984
            elif k == -2:
                w = x * x2m1 * (x - 2.0) * x2m9 * (x - 4.0) * 0.001388888889
            elif k == -1:
                w = x * (x - 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.004166666667
            elif k == 0:
                w = x2m1 * x2m4 * x2m9 * (x - 4.0) * 0.006944444444
            elif k == 1:
                w = x * (x + 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.006944444444
            elif k == 2:
                w = x * x2m1 * (x + 2.0) * x2m9 * (x - 4.0) * 0.004166666667
            elif k == 3:
                w = x * x2m1 * x2m4 * (x + 3.0) * (4.0 - x) * 0.001388888889
            elif k == 4:
                w = x * x2m1 * x2m4 * x2m9 * 0.0001984126984
        else:  # wsinc5: windowed sinc, raw (matches _interp_1d_along), taps -4..5
            d = tl.abs(x - k)
            pid_ = PIF * d
            s_ = tl.where(d > 0.01, tl.sin(pid_) / pid_, 1.0 - 1.6449341 * d * d)
            mxw = 0.19999 * d
            m3 = 0.4243801 + 0.4973406 * tl.cos(PIF * mxw) + 0.0782793 * tl.cos(PIF * mxw * 2.0)
            w = s_ * m3

        acc += w * val

    tl.store(outp + voff + idx, acc, mask=m)


def _unpack_ab(ax: Tensor, scl: Tensor) -> tuple[Tensor, Tensor]:
    """Vectorized fr_gpu_unpack: (ax,scl) -> per-step (a,b). ax (B,4), scl (B,4,3)."""
    a = torch.where(ax == 0, scl[..., 1], scl[..., 0])  # x:scl1 else scl0
    b = torch.where(ax == 0, scl[..., 2], torch.where(ax == 1, scl[..., 2], scl[..., 1]))
    return a, b


@triton.jit
def _phase_ramp_kernel(
    spec,  # float32 view of a complex64 (B, S0, S1, S2) spectrum, any strides
    a_ptr,
    b_ptr,
    s_ptr,
    S0,
    S1,
    S2,
    sb,  # strides in complex elements: rfft along a middle axis returns the
    s0,  # frequency axis innermost, and a .contiguous() copy of it cost more
    s1,  # than the ramp itself
    s2,
    n_fft,
    c1,
    c2,
    blocks_per_vol,
    K_DIM: tl.constexpr,
    P1_DIM: tl.constexpr,
    P2_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """spec *= exp(-2 pi i (k/n) (a (p1 - c1) + b (p2 - c2) + s)), in place.

    One pass over the spectrum with the ramp computed on the fly; building it
    as three broadcast factors cost three full complex passes (22 of 59 ms per
    58-volume 160x160x114 batch).
    """
    pid = tl.program_id(0)
    vol = pid // blocks_per_vol
    r = (pid % blocks_per_vol) * BLOCK + tl.arange(0, BLOCK)
    per_vol = S0 * S1 * S2
    m = r < per_vol
    i0 = r // (S1 * S2)
    i1 = (r // S2) % S1
    i2 = r % S2
    k = tl.where(K_DIM == 0, i0, tl.where(K_DIM == 1, i1, i2)).to(tl.float32)
    p1 = tl.where(P1_DIM == 0, i0, tl.where(P1_DIM == 1, i1, i2)).to(tl.float32)
    p2 = tl.where(P2_DIM == 0, i0, tl.where(P2_DIM == 1, i1, i2)).to(tl.float32)
    a = tl.load(a_ptr + vol)
    b = tl.load(b_ptr + vol)
    s = tl.load(s_ptr + vol)
    ang = -6.283185307179586 * (k / n_fft) * (a * (p1 - c1) + b * (p2 - c2) + s)
    # The inverse FFT's 1/n rides here: irfft(norm="forward") skips its own pass.
    cs = tl.cos(ang) / n_fft
    sn = tl.sin(ang) / n_fft
    base = (vol.to(tl.int64) * sb + i0.to(tl.int64) * s0 + i1 * s1 + i2 * s2) * 2
    re = tl.load(spec + base, mask=m)
    im = tl.load(spec + base + 1, mask=m)
    tl.store(spec + base, re * cs - im * sn, mask=m)
    tl.store(spec + base + 1, re * sn + im * cs, mask=m)


# fr-axis -> (tensor dim within (nz, ny, nx), p1 dim, p2 dim), as shear._AXIS_INFO
_FFT_DIMS = {0: (2, 1, 0), 1: (1, 2, 0), 2: (0, 2, 1)}


def _fourier_shears_shared(
    sources: Tensor,
    order: list[int],
    a4: Tensor,
    b4: Tensor,
    s4: Tensor,
    fft_axes: tuple[int, ...],
    mode_id: int,
) -> Tensor:
    """The four shears when every volume shares the axis order (``common`` plan):
    each step is one in-place FFT ramp or one Triton pass over the whole batch,
    with no per-axis gathers."""
    B, nz, ny, nx = sources.shape
    sizes = (nz, ny, nx)
    nxyz = nx * ny * nz
    BLOCK = 256
    blocks_per_vol = (nxyz + BLOCK - 1) // BLOCK
    half = {0: (nx - 1) / 2.0, 1: (ny - 1) / 2.0, 2: (nz - 1) / 2.0}
    p_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # fr ids of the two driving axes
    # AFNI skips a shear under 1e-3 voxel; a zero ramp is exactly that.
    st = torch.stack(
        [
            a4[:, i].abs() * half[p_axes[order[i]][0]]
            + b4[:, i].abs() * half[p_axes[order[i]][1]]
            + s4[:, i].abs()
            for i in range(4)
        ],
        dim=1,
    )
    live = (st >= 1.0e-3).to(torch.float32)
    a_l = (a4.to(torch.float32) * live).contiguous()
    b_l = (b4.to(torch.float32) * live).contiguous()
    s_l = (s4.to(torch.float32) * live).contiguous()
    ax_i = torch.tensor(order, dtype=torch.int32, device=sources.device).expand(B, 4).contiguous()
    cur = sources.contiguous()
    for step, v in enumerate(order):
        if v in fft_axes:
            kd, p1d, p2d = _FFT_DIMS[v]
            n = sizes[kd]
            spec = torch.fft.rfft(cur, dim=kd + 1)
            # Walk the spectrum in memory order (innermost stride last) so the
            # loads coalesce; logical (i0, i1, i2) indices follow the permutation.
            shp, std = spec.shape[1:], spec.stride()[1:]
            phys = sorted(range(3), key=lambda dd: -std[dd])
            S0, S1, S2 = (shp[dd] for dd in phys)
            s0, s1, s2 = (std[dd] for dd in phys)
            sb = spec.stride(0)
            kd_p, p1d_p, p2d_p = phys.index(kd), phys.index(p1d), phys.index(p2d)
            bpv = (S0 * S1 * S2 + BLOCK - 1) // BLOCK
            _phase_ramp_kernel[(B * bpv,)](
                torch.view_as_real(spec),
                a_l[:, step].contiguous(),
                b_l[:, step].contiguous(),
                s_l[:, step].contiguous(),
                S0,
                S1,
                S2,
                sb,
                s0,
                s1,
                s2,
                float(n),
                half[p_axes[v][0]],
                half[p_axes[v][1]],
                bpv,
                K_DIM=kd_p,
                P1_DIM=p1d_p,
                P2_DIM=p2d_p,
                BLOCK=BLOCK,
            )
            cur = torch.fft.irfft(spec, n=n, dim=kd + 1, norm="forward")
        else:
            # Explicitly contiguous: after a y-axis inverse FFT ``cur`` is strided,
            # empty_like would copy that layout, and out.reshape would then be a
            # copy the kernel writes into while ``out`` stays uninitialized.
            cur = cur.contiguous()
            out = torch.empty((B, nz, ny, nx), device=cur.device, dtype=cur.dtype)
            _shear_step_kernel[(B * blocks_per_vol,)](
                cur.reshape(B, nxyz),
                out.reshape(B, nxyz),
                ax_i,
                a_l,
                b_l,
                s_l,
                B,
                nx,
                ny,
                nz,
                blocks_per_vol,
                step,
                MODE=mode_id,
                BLOCK=BLOCK,
            )
            cur = out
    return cur


def _fourier_shears(
    sources: Tensor,
    ax: Tensor,
    a4: Tensor,
    b4: Tensor,
    s4: Tensor,
    fft_axes: tuple[int, ...],
    mode_id: int,
) -> Tensor:
    """Four shears, each one either the Triton polynomial or an FFT phase ramp.

    The plan's axis order differs per volume, so a step mixes both kinds. The
    Triton pass runs every volume with the FFT lanes' shear zeroed -- exactly
    the identity for an interpolating kernel -- and the FFT then overwrites
    those lanes from the step's input.
    """
    B, nz, ny, nx = sources.shape
    nxyz = nx * ny * nz
    BLOCK = 256
    blocks_per_vol = (nxyz + BLOCK - 1) // BLOCK
    on_fft = torch.zeros_like(ax, dtype=torch.bool)
    for v in fft_axes:
        on_fft |= ax == v
    zero = torch.zeros((), device=a4.device, dtype=a4.dtype)
    ax_i = ax.to(torch.int32).contiguous()
    a_p = torch.where(on_fft, zero, a4).to(torch.float32).contiguous()
    b_p = torch.where(on_fft, zero, b4).to(torch.float32).contiguous()
    s_p = torch.where(on_fft, zero, s4).to(torch.float32).contiguous()
    cur = sources.reshape(B, nxyz).contiguous()
    for step in range(4):
        out = torch.empty_like(cur)
        if bool((~on_fft[:, step]).any()):
            _shear_step_kernel[(B * blocks_per_vol,)](
                cur,
                out,
                ax_i,
                a_p,
                b_p,
                s_p,
                B,
                nx,
                ny,
                nz,
                blocks_per_vol,
                step,
                MODE=mode_id,
                BLOCK=BLOCK,
            )
        else:
            out.copy_(cur)
        vol_in = cur.reshape(B, nz, ny, nx)
        vol_out = out.reshape(B, nz, ny, nx)
        for v in fft_axes:
            lanes = (ax[:, step] == v).nonzero(as_tuple=True)[0]
            if lanes.numel():
                fft_shear_rows_batched(
                    vol_in, vol_out, lanes, v, a4[:, step], b4[:, step], s4[:, step]
                )
        cur = out
    return cur.reshape(B, nz, ny, nx)


def _fourier_grouped(sources, matrices, shape, ax, a4, b4, sft, valid, fft_axes, mode_id):
    """Fourier shears in groups that each share one axis order.

    The common plan rejects lanes it would condition badly; those form the
    next group with an order of their own (one or two groups in practice), so
    every factorizable lane stays batched. Re-solving them one volume at a
    time instead cost 6x the whole estimation. Lanes no order can factor
    (exact single-axis) come back invalid for the caller, as before.
    """
    B = sources.shape[0]
    out = sources.clone()
    done = torch.zeros(B, dtype=torch.bool, device=sources.device)
    lanes = torch.arange(B, device=sources.device)
    for _ in range(6):
        if not bool(valid.any()):
            break
        sel = lanes[valid]
        order = ax[valid][0].tolist()
        whole = sel.numel() == B and lanes.numel() == B
        src = sources if whole else sources[sel]
        res = _fourier_shears_shared(
            src, order, a4[valid], b4[valid], sft[valid], fft_axes, mode_id
        )
        if whole:
            return res, valid
        out[sel] = res
        done[sel] = True
        lanes = lanes[~valid]
        if lanes.numel() == 0:
            break
        ax, scl, sft, valid = rigid_matrix_to_shears(matrices[lanes], shape, common=True)
        a4, b4 = _unpack_ab(ax, scl)
    return out, done


def shear_resample_triton(
    sources: Tensor, matrices: Tensor, shape: tuple[int, int, int], mode: str
) -> tuple[Tensor, Tensor]:
    """Batched rigid resample via fused Triton shears.

    sources: (B,nz,ny,nx) on CUDA. matrices: (B,4,4) voxel-index pull matrices.
    Returns (out (B,nz,ny,nx), valid (B,)). Volumes with an invalid (degenerate)
    decomposition are returned unmodified (caller should overwrite them via a
    general resample).
    """
    B, nz, ny, nx = sources.shape
    fft_axes, poly = split_fourier_mode(mode)
    mode_id = _MODE_ID[poly]
    ax, scl, sft, valid = rigid_matrix_to_shears(matrices, shape, common=bool(fft_axes))
    a4, b4 = _unpack_ab(ax, scl)  # (B,4)
    if fft_axes:
        return _fourier_grouped(sources, matrices, shape, ax, a4, b4, sft, valid, fft_axes, mode_id)
    ax_i = ax.to(torch.int32).contiguous()
    a4 = a4.to(torch.float32).contiguous()
    b4 = b4.to(torch.float32).contiguous()
    s4 = sft.to(torch.float32).contiguous()

    nxyz = nx * ny * nz
    BLOCK = 256
    blocks_per_vol = (nxyz + BLOCK - 1) // BLOCK
    grid = (B * blocks_per_vol,)

    src = sources.reshape(B, nxyz)
    if not src.is_contiguous():
        src = src.contiguous()
    # Two fresh buffers ping-ponged; `src` is read-only (step 0 reads it, later
    # steps never write it) so we never mutate the caller's tensor.
    buf_a = torch.empty_like(src)
    buf_b = torch.empty_like(src)
    _shear_step_kernel[grid](
        src,
        buf_a,
        ax_i,
        a4,
        b4,
        s4,
        B,
        nx,
        ny,
        nz,
        blocks_per_vol,
        0,
        MODE=mode_id,
        BLOCK=BLOCK,
    )
    cur, other = buf_a, buf_b
    for step in (1, 2, 3):
        _shear_step_kernel[grid](
            cur,
            other,
            ax_i,
            a4,
            b4,
            s4,
            B,
            nx,
            ny,
            nz,
            blocks_per_vol,
            step,
            MODE=mode_id,
            BLOCK=BLOCK,
        )
        cur, other = other, cur
    return cur.reshape(B, nz, ny, nx), valid
