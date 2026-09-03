"""The streamed-Gram warp PCA must agree with an explicit float64 SVD.

``warp_pc_basis`` used to form the whole ``(T, S)`` matrix in float64 and call
``linalg.svd``; on a 440-frame 0.8mm run that was a 5 GB allocation and 30 s. It now
accumulates the ``(T, T)`` Gram over voxel chunks and eigendecomposes that instead.
Same subspace, same spectrum — these tests are what pins that down.
"""

import torch

from fastfuncstuff.processing.locomoco import (
    _warp_matrix,
    warp_pc_axis_bases,
    warp_pc_basis,
    warp_pc_reconstruct,
)

CPU = torch.device("cpu")
K = 5


def _components(seed=0, n_t=40, shape=(7, 6, 5), rank=4):
    """Two low-rank-plus-noise displacement fields, as ``warp_components`` returns."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for axis in (1, 2):
        n_vox = shape[0] * shape[1] * shape[2]
        a = torch.randn(n_t, rank, generator=g)
        b = torch.randn(rank, n_vox, generator=g)
        d = (a @ b).T.reshape(*shape, n_t) + 0.1 * torch.randn(*shape, n_t, generator=g)
        out.append((axis, d.contiguous()))
    return out


def _reference(components):
    """``(u, sv, xc, widths, shapes, mean)`` from the explicit float64 SVD."""
    x, mean, shapes, _axes, widths = _warp_matrix(components)
    xc = x - mean
    u, sv, _ = torch.linalg.svd(xc, full_matrices=False)
    return u, sv, xc, widths, shapes, mean


def test_matches_explicit_svd():
    comps = _components()
    u_ref, sv_ref, xc, widths, shapes, mean = _reference(comps)
    var_ref = (sv_ref[:K] ** 2) / (sv_ref**2).sum()

    u, loadings, means, var = warp_pc_basis(comps, n_pcs=K, device=CPU)

    assert torch.allclose(var, var_ref, rtol=1e-5)
    # Component signs are a convention, so compare the subspace directions.
    assert torch.allclose(
        (u.T @ u_ref[:, :K]).abs().diag(), torch.ones(K, dtype=u.dtype), atol=1e-6
    )

    sign = torch.sign((u * u_ref[:, :K]).sum(dim=0))
    load_ref = u_ref[:, :K].T @ xc
    start = 0
    for (_axis, load), w, shape in zip(loadings, widths, shapes, strict=True):
        want = load_ref[:, start : start + w].T.reshape(*shape, K) * sign
        assert torch.allclose(load, want, atol=1e-4 * float(want.abs().max()))
        start += w
    start = 0
    for mu, w, shape in zip(means, widths, shapes, strict=True):
        assert torch.allclose(mu, mean[0, start : start + w].reshape(shape), atol=1e-6)
        start += w


def test_sign_is_deterministic():
    """``eigh`` picks an arbitrary column sign; the basis must not flip between calls."""
    comps = _components(seed=3)
    a, _l, _m, _v = warp_pc_basis(comps, n_pcs=K, device=CPU)
    b, _l, _m, _v = warp_pc_basis(comps, n_pcs=K, device=CPU)
    # Not bit-identical: the chunk size is sized off free memory, so the Gram's
    # summation order can differ between calls. The SIGN must not.
    assert torch.allclose(a, b, atol=1e-8)
    assert (a[a.abs().argmax(dim=0), torch.arange(K)] > 0).all()


def test_balance_matches_the_old_formulas():
    comps = _components(seed=1)
    u_ref, _sv, xc, widths, _shapes, _mean = _reference(comps)
    u, _loadings, _means, _var, balance = warp_pc_basis(
        comps, n_pcs=K, device=CPU, with_balance=True, with_loadings=False
    )
    load_ref = u_ref[:, :K].T @ xc
    energy_all = (load_ref**2).sum(dim=1)
    start = 0
    for bal, w in zip(balance, widths, strict=True):
        blk = load_ref[:, start : start + w]
        tot = float((xc[:, start : start + w] ** 2).sum())
        per_k = (blk**2).sum(dim=1)
        solo = torch.linalg.svdvals(xc[:, start : start + w])[:K]
        assert abs(bal["energy"] - tot) < 1e-6 * tot
        assert torch.allclose(bal["share"], per_k / energy_all, atol=1e-6)
        assert torch.allclose(bal["shared_ev"], (per_k.cumsum(0) / tot).clamp(max=1.0), atol=1e-6)
        assert torch.allclose(bal["solo_ev"], ((solo**2).cumsum(0) / tot).clamp(max=1.0), atol=1e-6)
        start += w


def test_axis_bases_match_per_axis_svd():
    comps = _components(seed=2)
    _u, _sv, xc, widths, _shapes, _mean = _reference(comps)
    bases = warp_pc_axis_bases(comps, n_pcs=K, device=CPU)
    assert [axis for axis, _s, _v in bases] == [axis for axis, _d in comps]
    start = 0
    for (_axis, scores, var), w in zip(bases, widths, strict=True):
        blk = xc[:, start : start + w]
        u_ref, sv, _ = torch.linalg.svd(blk, full_matrices=False)
        assert torch.allclose(var, ((sv[:K] ** 2) / (sv**2).sum()).float(), rtol=1e-4)
        cos = (torch.nn.functional.normalize(scores.double(), dim=0).T @ u_ref[:, :K]).abs().diag()
        assert torch.allclose(cos, torch.ones(K, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(scores.std(dim=0), torch.ones(K), atol=1e-5)
        start += w


def test_without_loadings_returns_the_same_scores():
    comps = _components(seed=4)
    u_full, loadings, means, var_full = warp_pc_basis(comps, n_pcs=K, device=CPU)
    u_lean, empty_load, empty_mean, var_lean = warp_pc_basis(
        comps, n_pcs=K, device=CPU, with_loadings=False
    )
    assert torch.allclose(u_full, u_lean, atol=1e-8)
    assert torch.allclose(var_full, var_lean, atol=1e-10)
    assert loadings and means
    assert empty_load == [] and empty_mean == []


def test_full_rank_reconstruction_is_the_input():
    """Every component kept plus the restored mean is the field back, to float32."""
    comps = _components(seed=5)
    recon = warp_pc_reconstruct(comps, keep={}, n_pcs=None, device=CPU)
    for (axis, got), (axis_ref, want) in zip(recon, comps, strict=True):
        assert axis == axis_ref
        assert torch.allclose(got, want, atol=1e-4)


def test_input_fields_are_not_modified():
    """The streaming passes centre a COPY; on CPU the staged chunk IS the caller's data."""
    comps = _components(seed=6)
    before = [d.clone() for _axis, d in comps]
    warp_pc_basis(comps, n_pcs=K, device=CPU, with_balance=True)
    warp_pc_axis_bases(comps, n_pcs=K, device=CPU)
    for (_axis, after), want in zip(comps, before, strict=True):
        assert torch.equal(after, want)
