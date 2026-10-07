"""Smoothing along the cortex, and measuring how smooth data on a mesh are.

**Smoothing is heat diffusion** (Chung's heat-kernel smoothing; AFNI ``SurfSmooth
-met HEAT_07``): data diffuse along the surface for a time ``t``, never across a
sulcus. On a mesh that is ``M du/dt = -L u`` with ``L`` the cotangent Laplacian and
``M`` the lumped (vertex-area) mass matrix. The mass weighting is what makes the
result a property of the cortex rather than of the mesh: without it a dense patch of
vertices smooths less per step than a sparse one, the same lesson the surface editor
learned. Each step is implicit, ``(M + dt L) u' = M u``, so it is stable at any step size,
and the same factorisation serves every step and every time point. ``n`` steps of
``t / n`` converge to the Gaussian heat kernel, whose FWHM is ``sqrt(16 ln2 t)``. Fewer
steps keep the Gaussian's variance exactly (``2t`` per axis for any ``n``) but put more
of it in the tails, so the half-maximum width falls short: 0.89x at 8 steps, 0.95x at 16
(:func:`_step_fwhm_ratio`, a Hankel transform). ``t`` is scaled up to hit the target
FWHM, so the kernel's half-maximum width matches ``-fwhm`` the way a Gaussian blur's would.
Its variance then exceeds the Gaussian's by ``1 / ratio^2`` (11% at the default 16 steps,
25% at 8), and variances are what add when blurs compose: smoothing data that are already
3.4 mm smooth by 4 mm measured 5.67 at 8 steps against 5.22 in quadrature.
One step alone has a log singularity at its centre in 2-D, so at least 4 are taken.

Only edges inside the mask diffuse, so the medial wall and any vertex outside the EPI
neither lend nor take signal: the mask edge is a no-flux boundary and the mean of
the data inside it is conserved.

**Smoothness** is measured the way 3dFWHMx's classic estimator does on a grid, with
mesh edges in place of grid neighbours, and in its convention: the FWHM of the Gaussian
kernel that would make white noise this smooth. Two points ``d`` apart then correlate
at ``exp(-2 ln2 d^2 / FWHM^2)`` (the field's own correlation is that kernel convolved
with itself, sqrt(2) wider). The correlation of standardised
residuals across each edge, over time, gives one FWHM per edge (and so per vertex),
and a least-squares fit over every edge gives the global number. This is the mesh
analogue of SUMA's ``SurfFWHM``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .mesh import vertex_areas

__all__ = ["HeatSmoother", "cotan_laplacian", "mesh_edges", "surface_fwhm"]

_LN2 = float(np.log(2.0))


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E, 2), smaller index first."""
    f = np.asarray(faces, np.int64)
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    return np.unique(np.sort(e, axis=1), axis=0)


def cotan_laplacian(
    vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray | None = None
) -> sparse.csr_matrix:
    """The cotangent Laplacian ``L = D - W`` (positive semi-definite), as a sparse (V, V).

    ``W_ij = (cot a + cot b) / 2`` over the two angles facing edge ij. With ``mask``,
    edges that touch a vertex outside it carry no weight (a no-flux boundary).
    Negative cotangents (obtuse triangles) are clipped at 0: a negative weight can
    make a smoothing step create new extrema.
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    n = v.shape[0]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        a, b = v[i] - v[o], v[j] - v[o]
        cot = np.einsum("ij,ij->i", a, b) / np.maximum(
            np.linalg.norm(np.cross(a, b), axis=1), 1e-12
        )
        w = 0.5 * np.clip(cot, 0.0, None)
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    if mask is not None:
        m = np.asarray(mask, bool)
        keep = m[r] & m[c]
        r, c, w = r[keep], c[keep], w[keep]
    weights = sparse.csr_matrix((w, (r, c)), shape=(n, n))
    return (sparse.diags(np.asarray(weights.sum(axis=1)).ravel()) - weights).tocsr()


@lru_cache(maxsize=32)
def _step_fwhm_ratio(n_steps: int) -> float:
    """Half-max width of ``n`` implicit steps over the Gaussian's, same ``t`` (2-D)."""
    from scipy.optimize import brentq
    from scipy.special import j0

    dt = 1.0 / n_steps
    k = np.linspace(0.0, 60.0, 200001)
    spectrum = (1.0 + dt * k * k) ** (-n_steps) * k

    def kernel(r: float) -> float:
        return float(np.trapezoid(spectrum * j0(k * r), k))

    peak = kernel(0.0)
    half = brentq(lambda r: kernel(r) - 0.5 * peak, 1e-6, 20.0)
    return float(2.0 * half / np.sqrt(16.0 * _LN2))


class HeatSmoother:
    """Smooth per-vertex data to a target FWHM (mm) along a surface.

    ``vertices`` should be the **midthickness** of the brain the data came from (white
    is too wrinkled, inflated distorts distances, and a template's average surface is
    not this brain). Vertices outside ``mask`` are left untouched.

    Two solvers, one operator. With ``device=None`` each step is a sparse direct solve
    (one factorisation on the CPU, exact). With a torch ``device`` each step is a
    Jacobi-preconditioned conjugate-gradient solve over every column at once: the
    system ``M + dt L`` is mass-dominated and well conditioned (a few to a few tens of
    iterations), so on a GPU thousands of columns -- time points, or ClustSim noise
    realisations -- go through as one sparse-times-dense product per iteration.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        fwhm: float,
        mask: np.ndarray | None = None,
        n_steps: int = 16,
        device=None,
        tol: float = 1e-6,
    ):
        n = np.asarray(vertices).shape[0]
        self.mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        self.fwhm = float(fwhm)
        self.n_steps = max(int(n_steps), 4)
        ratio = _step_fwhm_ratio(self.n_steps)
        self.t = (self.fwhm / ratio) ** 2 / (16.0 * _LN2)
        idx = np.flatnonzero(self.mask)
        self._idx = idx
        lap = cotan_laplacian(vertices, faces, self.mask)[idx][:, idx]
        mass = vertex_areas(np.asarray(vertices, np.float64), np.asarray(faces, np.int64))[idx]
        # A masked vertex with no area left (every neighbour outside) keeps its value.
        self._mass = np.maximum(mass, 1e-12)
        dt = self.t / max(self.n_steps, 1)
        a = (sparse.diags(self._mass) + dt * lap).tocsr()
        self.device = device
        self.tol = float(tol)
        self._solve = None
        if self.fwhm > 0 and device is None:
            self._solve = spla.factorized(a.tocsc())
        elif self.fwhm > 0:
            import warnings

            import torch

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
                self._a = torch.sparse_csr_tensor(
                    torch.as_tensor(a.indptr, dtype=torch.int64),
                    torch.as_tensor(a.indices, dtype=torch.int64),
                    torch.as_tensor(a.data, dtype=torch.float32),
                    size=a.shape,
                    check_invariants=False,
                ).to(device)
            self._m_t = torch.as_tensor(self._mass, dtype=torch.float32, device=device)
            self._dinv = 1.0 / torch.as_tensor(a.diagonal(), dtype=torch.float32, device=device)

    def __call__(self, data):
        """``(V,)`` or ``(V, T)`` -> the same, smoothed inside the mask.

        numpy in, numpy out; a torch tensor in (with a ``device`` smoother), a tensor
        out on that device.
        """
        if self.device is not None:
            return self._call_torch(data)
        x = np.asarray(data)
        out = np.array(x, dtype=np.float64 if x.dtype == np.float64 else np.float32, copy=True)
        if self._solve is None:
            return out
        u = x[self._idx].astype(np.float64)
        m = self._mass if u.ndim == 1 else self._mass[:, None]
        for _ in range(self.n_steps):
            u = self._solve(m * u)
        out[self._idx] = u
        return out

    def _call_torch(self, data):
        import torch

        was_numpy = not isinstance(data, torch.Tensor)
        x = torch.as_tensor(np.asarray(data) if was_numpy else data, dtype=torch.float32)
        x = x.to(self.device)
        out = x.clone()
        if self.fwhm <= 0:
            return out.cpu().numpy() if was_numpy else out
        idx = torch.as_tensor(self._idx, device=self.device)
        u = x[idx]
        vec = u.ndim == 1
        if vec:
            u = u[:, None]
        for _ in range(self.n_steps):
            u = self._cg(self._m_t[:, None] * u, u)
        out[idx] = u[:, 0] if vec else u
        return out.cpu().numpy() if was_numpy else out

    def _cg(self, b, x0):
        """Solve ``A X = B`` column-wise by Jacobi-preconditioned CG, from ``x0``."""
        import torch

        x = x0.clone()
        r = b - self._a @ x
        z = self._dinv[:, None] * r
        p = z.clone()
        rz = (r * z).sum(0)
        bnorm = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-30)
        for _ in range(500):
            ap = self._a @ p
            alpha = rz / (p * ap).sum(0).clamp_min(1e-30)
            x += alpha * p
            r -= alpha * ap
            if bool((torch.linalg.vector_norm(r, dim=0) / bnorm).max() < self.tol):
                break
            z = self._dinv[:, None] * r
            rz_new = (r * z).sum(0)
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        return x


def surface_fwhm(
    residuals: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """``(global FWHM, per-vertex FWHM)`` in mm along the surface from ``(V, T)`` residuals.

    Each edge inside the mask gives the correlation of the standardised residuals at
    its two ends; the global value is the least-squares fit of
    ``ln corr = -2 ln2 d^2 / FWHM^2`` (3dFWHMx's kernel convention) over every edge, the per-vertex value the same fit
    over the edges at that vertex (NaN where there is none). Edges whose correlation is
    not in (0, 1) carry no information about a Gaussian width and are skipped.
    """
    r = np.asarray(residuals, np.float64)
    v = np.asarray(vertices, np.float64)
    n = v.shape[0]
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    z = r - r.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    m &= sd > 0
    z = np.divide(z, sd[:, None], out=np.zeros_like(z), where=sd[:, None] > 0)
    e = mesh_edges(faces)
    e = e[m[e[:, 0]] & m[e[:, 1]]]
    corr = np.einsum("et,et->e", z[e[:, 0]], z[e[:, 1]]) / z.shape[1]
    d2 = np.sum((v[e[:, 0]] - v[e[:, 1]]) ** 2, axis=1)
    ok = (corr > 0) & (corr < 1)
    e, corr, d2 = e[ok], corr[ok], d2[ok]
    y = -np.log(corr)  # = 2 ln2 d^2 / FWHM^2
    # 1/FWHM^2 = sum(y d^2) / (2 ln2 sum(d^4)), the least-squares slope through 0.
    inv = (y * d2).sum() / (2 * _LN2 * (d2 * d2).sum()) if len(y) else np.nan
    fwhm = float(1.0 / np.sqrt(inv)) if inv > 0 else float("nan")
    num = np.bincount(e.ravel(), np.repeat(y * d2, 2), minlength=n)
    den = np.bincount(e.ravel(), np.repeat(d2 * d2, 2), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = 1.0 / np.sqrt(num / (2 * _LN2 * den))
    per[(den == 0) | ~(num > 0)] = np.nan
    return fwhm, per
