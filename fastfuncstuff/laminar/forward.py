"""
The laminar BOLD generative model: state equations and observation equation.

Chain: neuronal excitatory/inhibitory activity at ``N`` depths -> neurovascular
coupling -> CBF -> a venule + ascending-vein compartment network at ``K``
vascular depths -> the laminar BOLD signal equation -> a depth point-spread
function that maps model depths onto sampled voxels.

Everything here is **batch-first**. Parameters carry leading batch dimensions
and broadcast against the state tensor, so a whole grid of (model x vascular
resolution x subject x restart) evaluates in one call. That is the only axis on
which a GPU helps: a single 50-state ODE is launch-bound and runs faster on the
CPU (see ``concepts/Laminar generative model.md``).

State layout (last axis, matching the reference's column-major ``[xn(:); xk(:)]``
so that vectors and Jacobians line up with the MATLAB oracle):

===============  ==========================  ==============
slice            quantity                    scale
===============  ==========================  ==============
``[0:N]``        excitatory activity         linear
``[N:2N]``       inhibitory activity         linear
``[2N:3N]``      vasoactive signal           linear
``[3N:4N]``      CBF                         log
``[4N+0K:4N+1K]``  venule blood volume       log
``[4N+1K:4N+2K]``  venule dHb                log
``[4N+2K:4N+3K]``  ascending-vein volume     log
``[4N+3K:4N+4K]``  ascending-vein dHb        log
===============  ==========================  ==============

Depth index 0 is **superficial** (closest to CSF); index K-1 is deepest. Blood
in the ascending vein flows from high index to low index.

References
----------
Havlicek M & Uludag K (2020). NeuroImage 204:116209.
Uludag K & Havlicek M (2021). Prog Neurobiol 207:102055.
Faes LK et al. (2026). Nat Commun. doi:10.1038/s41467-026-73540-z

Ported from ``LBR_gen_fx_fcn.m`` (which is the readable symbolic source the
``sym_fx_LBR_N*K*.m`` files are generated from), ``LBR_model_fx.m`` and
``LBR_model_gx.m``.
"""

from __future__ import annotations

import math

import torch

from fastfuncstuff.laminar.params import ModelSpec, p0_tensors


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def neuronal_to_vascular(
    spec: ModelSpec,
    P: dict[str, torch.Tensor],
    *,
    shift_depths: bool = False,
) -> torch.Tensor:
    """Gaussian basis mapping the N neuronal depths onto the K vascular depths.

    Returns ``(..., K, N)``, rows normalised to sum to 1 *before* the edge
    correction ``nb``. This is the piece that lets N and K be chosen
    independently -- Uludag & Havlicek (2021) recommend K = 7-10 for N = 3.

    ``shift_depths`` reproduces the reference's ``if P.s ~= 0`` branch, which
    skews the neuronal depth centres. Note the branch is *not* a no-op at
    ``P.s = 0``: it adds ``0.01 * (m - 0.5)``. Every published application fixes
    ``P.s = 0`` with zero prior variance, so the default is off.
    """
    N, K = spec.N, spec.K
    kw = {"dtype": spec.dtype, "device": spec.device}

    # Neuronal depth centres: the average of the "bin midpoint" and the
    # "interior node" conventions, then padded with one phantom depth on each
    # side so the Gaussians have somewhere to leak to.
    m1 = torch.linspace(0, 1, 2 * N + 1, **kw)[1::2]
    m2 = torch.linspace(0, 1, N + 2, **kw)[1:-1]
    m = (m1 + m2) / 2
    dm = m[1] - m[0]
    m = torch.cat([m[:1] - dm, m, m[-1:] + dm])  # (N+2,)
    m = m.expand(K + 2, N + 2)

    if shift_depths:
        m = m + 0.01 * (m - 0.5) * torch.exp(P["s"][..., None, None])

    # Width: fixed by N, with the estimated `nsig` moving it +/-20% through a
    # probit so the parameter stays unconstrained on the real line.
    base = (1.0 / (3 * (N + 1))) ** 2
    nsig = (base * 1.2 - base * 0.8) * _std_normal_cdf(P["nsig"][..., None, None]) + base * 0.8

    sp = torch.linspace(0, 1, 2 * K + 1, **kw)[1::2]
    dsp = sp[1] - sp[0]
    sp = torch.cat([sp[:1] - dsp, sp, sp[-1:] + dsp])  # (K+2,)
    sp = sp[:, None].expand(K + 2, N + 2)

    n2k = torch.rsqrt(2 * torch.pi * nsig) * torch.exp(-((sp - m) ** 2) / (2 * nsig))
    n2k = n2k / n2k.sum(dim=-1, keepdim=True)
    n2k = n2k[..., 1:-1, 1:-1]  # drop the phantom depths -> (..., K, N)

    # Boundary depths get extra weight from their own neuronal depth: the
    # phantom columns absorbed mass that physically has nowhere else to go.
    nb = 0.05 * torch.exp(P["nb"])
    corner = torch.zeros_like(n2k)
    corner[..., 0, 0] = 1.0
    corner[..., -1, -1] = 1.0
    return n2k + corner * nb[..., None, None]


def baseline_hemodynamics(spec: ModelSpec, P: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Depth profiles of baseline CBV, CBF and transit time.

    Model complexity does not grow with K: the depth profile of CBV0 is a
    single *slope* per compartment (``s_v``, ``s_d``), not K free numbers. This
    is what makes fitting the same parameter set at K = 7, 9, 10, 11 and
    averaging across them coherent.
    """
    p0 = p0_tensors(spec)
    depths = spec.depths.flip(0)  # descending: largest value at the surface

    V0t = p0["V0t"] * torch.exp(P["V0t"])
    w_v = p0["w_v"] * torch.exp(P["w_v"])
    w_d = 1.0 - w_v

    s_v = p0["s_v"] * torch.exp(P["s_v"])
    s_d = p0["s_d"] * torch.exp(P["s_d"])
    s_d0 = p0["s_d0"] * torch.exp(P["s_d0"])
    s_d2 = p0["s_d2"] * torch.exp(P["s_d2"])

    x_v = 10.0 + s_v[..., None] * depths
    # The reference's fx uses a hinge here and its gx uses a plain line; with
    # the default s_d2 = 0 the two agree. We follow fx.
    x_d = (
        10.0
        + s_d[..., None] * depths
        + s_d2[..., None] * torch.clamp(depths - s_d0[..., None], min=0.0)
    )
    x_v = x_v / x_v.sum(dim=-1, keepdim=True)
    x_d = x_d / x_d.sum(dim=-1, keepdim=True)

    V0v = V0t[..., None] * w_v[..., None] * x_v
    V0d = V0t[..., None] * w_d[..., None] * x_d

    t0v_par = p0["t0v"] * torch.exp(P["t0v"])
    F0v = V0v / t0v_par[..., None]
    # Flow accumulates toward the surface: F0d[k] = sum of venule outflow at
    # every depth at or below k. Note this makes F0d[K-1] == F0v[K-1] exactly,
    # which is why the deepest depth needs no special case in the ODE.
    F0d = torch.cumsum(F0v.flip(-1), dim=-1).flip(-1)

    return {
        "V0v": V0v,
        "V0d": V0d,
        "F0v": F0v,
        "F0d": F0d,
        "t0v": V0v / F0v,
        "t0d": V0d / F0d,
    }


def _expand_K(spec: ModelSpec, p0_val: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (p0_val * torch.exp(p))[..., None].expand(*p.shape, spec.K)


def _expand_N(spec: ModelSpec, p0_val: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (p0_val * torch.exp(p))[..., None].expand(*p.shape, spec.N)


def f_ode(
    x: torch.Tensor,
    u: torch.Tensor,
    P: dict[str, torch.Tensor],
    spec: ModelSpec,
    *,
    shift_depths: bool = False,
) -> torch.Tensor:
    """Right-hand side of the laminar generative model.

    Parameters
    ----------
    x : ``(..., 4N + 4K)`` state vector, see the module docstring for layout.
    u : ``(..., n_inputs)`` input at this instant. Modulatory inputs come
        first, then driving inputs -- ``B[m]`` multiplies ``u[..., m]``.
    P : estimated log-deviations from ``P0``.

    Notes
    -----
    The reference switches the viscoelastic time constant between inflation and
    deflation using the sign of ``dv/dt`` carried over from the *previous* call
    in a MATLAB ``persistent``, which makes its right-hand side depend on
    integration history. Faes et al. set ``tau_*_same = 1``, disabling the
    branch, and we only implement that case -- a history-dependent RHS is not a
    vector field and cannot be batched or differentiated cleanly.
    """
    if not (spec.p0.tau_v_same and spec.p0.tau_d_same):
        raise NotImplementedError(
            "asymmetric inflation/deflation tau needs the reference's history-dependent "
            "branch; set tau_v_same and tau_d_same"
        )

    p0 = p0_tensors(spec)
    N, K = spec.N, spec.K

    xE = x[..., 0:N]
    xI = x[..., N : 2 * N]
    xA = x[..., 2 * N : 3 * N]
    xF = torch.exp(x[..., 3 * N : 4 * N])

    base = 4 * N
    v_v = torch.exp(x[..., base + 0 * K : base + 1 * K])
    q_v = torch.exp(x[..., base + 1 * K : base + 2 * K])
    v_d = torch.exp(x[..., base + 2 * K : base + 3 * K])
    q_d = torch.exp(x[..., base + 3 * K : base + 4 * K])

    # ---- neuronal ---------------------------------------------------------
    # The diagonal of A is replaced by -sigma*exp(diag(A)): self-inhibition is
    # parameterised multiplicatively around the global time constant sigma, so
    # sigma alone sets the overall temporal scale of the neuronal response.
    sigma = p0["sigma"] * torch.exp(P["sigma"])
    A = P["A"]
    diagA = torch.diagonal(A, dim1=-2, dim2=-1)
    A = A - torch.diag_embed(diagA) - torch.diag_embed(sigma[..., None] * torch.exp(diagA))
    n_mod = P["B"].shape[-3]
    for m in range(n_mod):
        A = A + u[..., m, None, None] * P["B"][..., m, :, :]

    u_mod = u[..., :n_mod]
    mu = p0["mu"] * torch.exp(P["mu"][..., None] + (P["Bmu"] * u_mod[..., None, :]).sum(-1))
    lam = p0["lam"] * torch.exp(P["lam"][..., None] + (P["Blam"] * u_mod[..., None, :]).sum(-1))
    mu = mu.expand(*torch.broadcast_shapes(mu.shape[:-1], xE.shape[:-1]), N)
    lam = lam.expand(*torch.broadcast_shapes(lam.shape[:-1], xE.shape[:-1]), N)

    CU = (P["C"] * u[..., None, :]).sum(-1)  # (..., N)

    c1 = _expand_N(spec, p0["c1"], P["c1"])
    c2 = _expand_N(spec, p0["c2"], P["c2"])
    c3 = _expand_N(spec, p0["c3"], P["c3"])

    dxE = (A @ xE[..., None])[..., 0] - mu * xI + CU
    dxI = lam * (xE - xI)
    dxA = xE - c1 * xA
    dlog_xF = (c2 * xA - c3 * (xF - 1.0)) / xF

    # ---- neurovascular coupling across depths -----------------------------
    n2k = neuronal_to_vascular(spec, P, shift_depths=shift_depths)
    # Relative CBF at each vascular depth: a weighted blend of the neuronal
    # depths' CBF, expressed as a deviation from baseline so weights that do
    # not sum to 1 (after the nb correction) cannot shift the resting state.
    cbf_k = (n2k @ (xF - 1.0)[..., None])[..., 0] + 1.0

    # ---- hemodynamics -----------------------------------------------------
    hemo = baseline_hemodynamics(spec, P)
    F0v, F0d = hemo["F0v"], hemo["F0d"]
    t0v, t0d = hemo["t0v"], hemo["t0d"]

    al_v = _expand_K(spec, p0["al_v"], P["al_v"])
    al_d = _expand_K(spec, p0["al_d"], P["al_d"])
    nr = _expand_K(spec, p0["nr"], P["nr"])
    tau_v = _expand_K(spec, p0["tau_v_in"], P["tau_v_in"])
    tau_d = _expand_K(spec, p0["tau_d_in"], P["tau_d_in"])

    # Venules: outflow is the steady-state power law blended with the inflow by
    # the viscoelastic constant (Eq. 3 of Havlicek & Uludag 2020, rearranged).
    fv = (t0v * v_v ** (1.0 / al_v) + tau_v * cbf_k) / (t0v + tau_v)
    dlog_v_v = (cbf_k - fv) / (t0v * v_v)
    # CMRO2 from the n-ratio: m = (f - 1)/n + 1.
    cmro2 = (cbf_k + nr - 1.0) / nr
    dlog_q_v = (cmro2 - fv * q_v / v_v) / (t0v * q_v)

    # Ascending vein: a serial chain from the deepest depth up to the surface.
    # K is 7-11, so the Python loop is negligible next to the batch dimension.
    w_v_in = F0v / F0d  # venule share of this depth's AV inflow
    w_d_in = torch.zeros_like(F0d)
    w_d_in[..., :-1] = F0d[..., 1:] / F0d[..., :-1]  # share from the depth below

    dlog_v_d = torch.zeros_like(v_d)
    dlog_q_d = torch.zeros_like(q_d)
    fd_next = torch.zeros_like(v_d[..., 0])
    qv_ratio = q_v / v_v
    qd_ratio = q_d / v_d
    for k in range(K - 1, -1, -1):
        inflow = fv[..., k] * w_v_in[..., k] + fd_next * w_d_in[..., k]
        fd_k = (t0d[..., k] * v_d[..., k] ** (1.0 / al_d[..., k]) + tau_d[..., k] * inflow) / (
            t0d[..., k] + tau_d[..., k]
        )
        dhb_in = fv[..., k] * w_v_in[..., k] * qv_ratio[..., k]
        if k < K - 1:
            dhb_in = dhb_in + fd_next * w_d_in[..., k] * qd_ratio[..., k + 1]
        dlog_v_d[..., k] = (inflow - fd_k) / (t0d[..., k] * v_d[..., k])
        dlog_q_d[..., k] = (dhb_in - fd_k * qd_ratio[..., k]) / (t0d[..., k] * q_d[..., k])
        fd_next = fd_k

    return torch.cat([dxE, dxI, dxA, dlog_xF, dlog_v_v, dlog_q_v, dlog_v_d, dlog_q_d], dim=-1)


def g_obs(
    x: torch.Tensor,
    P: dict[str, torch.Tensor],
    spec: ModelSpec,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Laminar BOLD signal (percent change) at each of the K vascular depths.

    Eq. 6 of Havlicek & Uludag (2020): a CBV0-weighted sum of extravascular
    dHb-content, intravascular dHb-concentration and CBV terms. Field strength,
    TE and sequence enter only through the three ``k1/k2/k3`` scalars, so
    retargeting from 7T GE is a constant swap.

    ``kernel`` applies the depth point-spread function that maps model depths
    onto sampled voxels.
    """
    p0 = p0_tensors(spec)
    N, K = spec.N, spec.K
    base = 4 * N
    v_v = torch.exp(x[..., base + 0 * K : base + 1 * K])
    q_v = torch.exp(x[..., base + 1 * K : base + 2 * K])
    v_d = torch.exp(x[..., base + 2 * K : base + 3 * K])
    q_d = torch.exp(x[..., base + 3 * K : base + 4 * K])

    hemo = baseline_hemodynamics(spec, P)
    # CBV0 as a fraction of tissue rather than an absolute volume in mL.
    V0vq = hemo["V0v"] / 100.0 * K
    V0dq = hemo["V0d"] / 100.0 * K

    E0v = _expand_K(spec, p0["E0v"], P["E0v"])
    E0d = _expand_K(spec, p0["E0d"], P["E0d"])

    TE, B0 = spec.TE, spec.B0
    nu0v = p0["suscep"] * p0["gyro"] * p0["Hct_v"] * B0
    nu0d = p0["suscep"] * p0["gyro"] * p0["Hct_d"] * B0

    # Baseline intra-to-extra-vascular signal ratio
    ep_v = p0["rho_v"] / p0["rho_t"] * torch.exp(-TE * (p0["R2s_v"] - p0["R2s_t"]))
    ep_d = p0["rho_d"] / p0["rho_t"] * torch.exp(-TE * (p0["R2s_d"] - p0["R2s_t"]))

    H0 = 1.0 / (1.0 - V0vq - V0dq + ep_v * V0vq + ep_d * V0dq)

    k1v = 4.3 * nu0v * E0v * TE
    k2v = ep_v * p0["r0v"] * E0v * TE
    k3v = 1.0 - ep_v
    k1d = 4.3 * nu0d * E0d * TE
    # Bug of record: the reference scales the ascending vein's intravascular
    # term by the *venule* ratio ep_v. Replicated deliberately -- changing it
    # would silently break parity with the published fits.
    k2d = ep_v * p0["r0d"] * E0d * TE
    k3d = 1.0 - ep_d

    lbr = (
        H0
        * (
            (1.0 - V0vq - V0dq) * (k1v * V0vq * (1.0 - q_v) + k1d * V0dq * (1.0 - q_d))
            + k2v * V0vq * (1.0 - q_v / v_v)
            + k2d * V0dq * (1.0 - q_d / v_d)
            + k3v * V0vq * (1.0 - v_v)
            + k3d * V0dq * (1.0 - v_d)
        )
        * 100.0
    )

    if kernel is not None:
        lbr = apply_depth_psf(lbr, kernel)
    return lbr


def apply_depth_psf(lbr: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Blur the depth profile by the voxel point-spread function.

    The reference convolves the profile in both depth orientations, each padded
    by one replicated edge sample, and takes the elementwise maximum. Taking
    the max rather than one orientation keeps the blur from eating the profile's
    peak; it is asymmetric on purpose, since the physical leakage is toward the
    surface. Replicated literally -- this sits directly between model and data.
    """
    L = kernel.shape[-1]
    if L % 2 == 0:
        raise ValueError(f"depth PSF kernel must have odd length, got {L}")

    def _same_conv(a: torch.Tensor) -> torch.Tensor:
        # MATLAB conv(a, k, 'same') == correlation with the reversed kernel,
        # cropped to a's length.
        w = kernel.flip(-1).reshape(1, 1, L).to(a.dtype)
        flat = a.reshape(-1, 1, a.shape[-1])
        out = torch.nn.functional.conv1d(flat, w, padding=L // 2)
        return out.reshape(a.shape)

    # Pad with one replicated sample at each end before blurring so the edge
    # depths are not pulled toward zero, then drop the pad.
    def _padded(a: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([a[..., :1], a, a[..., -1:]], dim=-1)
        return _same_conv(padded)[..., 1:-1]

    up = _padded(lbr.flip(-1)).flip(-1)
    down = _padded(lbr)
    return torch.maximum(up, down)
