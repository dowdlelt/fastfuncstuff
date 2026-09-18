"""
Parity for the full eight-model space and the Bayesian parameter average.

``test_laminar_inversion.py`` checks two models at one vascular resolution.
This checks all eight at two, plus ``spm_dcm_bpa`` across them -- the parts of
the published pipeline that sit between a single inversion and the reported
result, and that nothing else verifies.

Note the kernel: the reference driver estimates the PSF once at K=7 and applies
that same seven-tap kernel at *every* K. Reproduced here, because the point is
to match what they ran. (The notebook deliberately estimates a kernel per K,
which is better and therefore not comparable.)

Regenerate with::

    cd tests/laminar_oracle && matlab -batch "run('make_modelspace_oracle.m')"

That takes ~40 minutes: MATLAB averages ~150 s per inversion against our ~21 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.experiment import (
    bayesian_parameter_average,
    faes_priors,
    layer_model_names,
    layer_model_targets,
    posterior_model_probabilities,
)
from fastfuncstuff.laminar.inversion import variational_laplace, vec
from fastfuncstuff.laminar.params import P0, ModelSpec

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_modelspace_oracle.json"


def _oracle():
    if not ORACLE.exists():  # pragma: no cover - only when the dump is missing
        pytest.skip(f"no model-space oracle at {ORACLE}")
    return json.loads(ORACLE.read_text())


def _spec(o, K):
    return ModelSpec(
        N=int(o["N"]),
        K=K,
        n_inputs=2,
        n_mod=1,
        dt=float(o["TR"]) / 32.0,
        TR=float(o["TR"]),
        p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5),
    )


_FIT_CACHE: dict[tuple[int, int], tuple] = {}


def _fit(o, ki, mi):
    """Invert model ``mi`` at the ``ki``-th vascular resolution.

    Cached: the tests below ask for the same 16 inversions from several angles,
    and at ~21 s each an uncached run would be a quarter-hour of refitting.
    """
    if (ki, mi) in _FIT_CACHE:
        return _FIT_CACHE[(ki, mi)]
    block = o["per_k"][ki]
    K = int(block["K"])
    spec = _spec(o, K)
    ns = int(block["ns"])

    def t(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)

    y = t(block["y"]).reshape(ns, K)
    u = t(block["u"]).reshape(-1, 2)
    rows = t(block["mask_rows"]).reshape(-1).to(torch.bool)
    kernel = t(o["kernel"]).reshape(-1)

    target = tuple(int(v) for v in np.asarray(o["targets"][mi]).ravel())
    priors = faes_priors(spec, target)
    res = variational_laplace(spec, priors, u, y, rows=rows, kernel=kernel)
    _FIT_CACHE[(ki, mi)] = (spec, res, block["models"][mi])
    return _FIT_CACHE[(ki, mi)]


# --------------------------------------------------------------------------
# The model space
# --------------------------------------------------------------------------


def test_model_space_matches_the_reference_ordering():
    """Our names and targets must line up with the oracle's, or every
    model-indexed comparison below is silently comparing the wrong pair."""
    o = _oracle()
    assert layer_model_names(3) == list(o["names"])
    ours = layer_model_targets(3)
    theirs = [tuple(int(v) for v in np.asarray(t).ravel()) for t in o["targets"]]
    assert ours == theirs


@pytest.mark.slow
@pytest.mark.parametrize("ki", [0, 1])
@pytest.mark.parametrize("mi", range(8))
def test_free_energy_matches_matlab_for_every_model(ki, mi):
    o = _oracle()
    _, res, ref = _fit(o, ki, mi)
    expect = float(ref["F"])
    rel = abs(float(res.F) - expect) / abs(expect)
    assert rel < 1e-6, (
        f"K={o['per_k'][ki]['K']} {ref['name']}: F={float(res.F):.6f} vs {expect:.6f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("ki", [0, 1])
def test_full_model_ranking_and_probabilities_match(ki):
    """The scientific output: which hypothesis wins, and by how much."""
    o = _oracle()
    ours, theirs = [], []
    for mi in range(8):
        _, res, ref = _fit(o, ki, mi)
        ours.append(float(res.F))
        theirs.append(float(ref["F"]))
    assert np.argmax(ours) == np.argmax(theirs)
    assert list(np.argsort(ours)) == list(np.argsort(theirs))
    p_ours = posterior_model_probabilities(ours).numpy()
    p_theirs = posterior_model_probabilities(theirs).numpy()
    assert np.abs(p_ours - p_theirs).max() < 1e-6


@pytest.mark.slow
def test_posterior_modulation_matches_for_every_model():
    """The estimated B per depth, which is what a result actually reports."""
    o = _oracle()
    for mi in range(8):
        _, res, ref = _fit(o, 1, mi)
        ours = torch.diagonal(res.Ep["B"][0]).numpy()
        theirs = np.asarray(ref["Ep_B"], dtype=float).reshape(-1)
        assert np.abs(ours - theirs).max() < 1e-6, ref["name"]


# --------------------------------------------------------------------------
# Bayesian parameter averaging across K
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_bpa_posterior_means_match_spm():
    """spm_dcm_bpa across K=7 and K=9, for the winning model."""
    o = _oracle()
    mi = 7  # superficial+middle+deep, the model that wins
    runs, specs = [], []
    for ki in (0, 1):
        spec, res, _ = _fit(o, ki, mi)
        runs.append(res)
        specs.append(spec)
    bpa = bayesian_parameter_average(runs, specs[0])

    ref = o["bpa"][mi]
    theirs_B = np.asarray(ref["Ep_B"], dtype=float).reshape(-1)
    ours_B = torch.diagonal(bpa.Ep["B"][0]).numpy()
    assert np.abs(ours_B - theirs_B).max() < 1e-6, f"B: {ours_B} vs {theirs_B}"


@pytest.mark.slow
def test_bpa_free_energy_is_the_first_k_not_the_sum():
    """Bug of record, now confirmed against SPM on real posteriors.

    ``spm_dcm_bpa`` sets ``BPA = DCM`` from the first DCM and never touches
    ``BPA.F``, despite its caller's comment saying the free energy is
    "accumulated over different number of BOLD depths". So the F that gets
    compared is the *first* K's. The oracle shows this directly: every model's
    BPA F equals its K=7 F exactly.
    """
    o = _oracle()
    for mi in range(8):
        bpa_F = float(o["bpa"][mi]["F"])
        k7_F = float(o["per_k"][0]["models"][mi]["F"])
        k9_F = float(o["per_k"][1]["models"][mi]["F"])
        assert bpa_F == pytest.approx(k7_F, rel=1e-12), o["names"][mi]
        assert bpa_F != pytest.approx(k7_F + k9_F, rel=1e-6)

    # And our default reproduces it, while the documented-but-unimplemented
    # behaviour is available explicitly.
    o2 = _oracle()
    mi = 3
    runs, specs = [], []
    for ki in (0, 1):
        spec, res, _ = _fit(o2, ki, mi)
        runs.append(res)
        specs.append(spec)
    ref_mode = bayesian_parameter_average(runs, specs[0], free_energy="reference")
    sum_mode = bayesian_parameter_average(runs, specs[0], free_energy="sum")
    assert float(ref_mode.F) == pytest.approx(float(runs[0].F), rel=1e-12)
    assert float(sum_mode.F) == pytest.approx(float(runs[0].F) + float(runs[1].F), rel=1e-12)
    assert float(ref_mode.F) == pytest.approx(float(o2["bpa"][mi]["F"]), rel=1e-6)


@pytest.mark.slow
def test_bpa_is_sharper_than_either_input():
    """Precision-weighted averaging must reduce posterior variance, not inflate
    it -- the property that makes averaging across K worth doing at all."""
    o = _oracle()
    runs, specs = [], []
    for ki in (0, 1):
        spec, res, _ = _fit(o, ki, 7)
        runs.append(res)
        specs.append(spec)
    bpa = bayesian_parameter_average(runs, specs[0])
    combined = torch.diagonal(bpa.Cp)
    for r in runs:
        assert torch.all(combined <= torch.diagonal(r.Cp) + 1e-12)


def test_reference_reuses_one_kernel_at_every_k():
    """A property of their pipeline worth pinning, because it is surprising.

    The PSF is described as adjusted for the number of vascular depths, but the
    driver estimates it once at K=7 and applies that seven-tap kernel at every K.
    Any comparison against published numbers has to do the same.
    """
    o = _oracle()
    kernel = np.asarray(o["kernel"], dtype=float).reshape(-1)
    assert kernel.size == 7
    assert [int(b["K"]) for b in o["per_k"]] == [7, 9]
    assert kernel.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Marginal depth inclusion (the recommended readout)
# --------------------------------------------------------------------------


def test_depth_inclusion_marginalises_over_the_model_space():
    """P(depth modulated) sums every model containing that depth."""
    from fastfuncstuff.laminar.experiment import depth_inclusion_probabilities

    targets = layer_model_targets(3)
    # A model space where 'deep' alone is overwhelmingly favoured.
    F = [-100.0] * 8
    F[targets.index((0, 0, 1))] = -50.0
    inc = depth_inclusion_probabilities(F).numpy()
    assert inc[2] > 0.99 and inc[0] < 0.01 and inc[1] < 0.01

    # A tie between 'deep' and 'superficial+deep' must leave deep certain and
    # superficial at one half -- the case winner-takes-all would report as a
    # categorical two-depth effect.
    F = [-1e6] * 8
    F[targets.index((0, 0, 1))] = 0.0
    F[targets.index((1, 0, 1))] = 0.0
    inc = depth_inclusion_probabilities(F).numpy()
    assert inc[2] == pytest.approx(1.0, abs=1e-9)
    assert inc[0] == pytest.approx(0.5, abs=1e-9)
    assert inc[1] == pytest.approx(0.0, abs=1e-9)


def test_depth_inclusion_is_bounded_and_matches_a_certain_model():
    from fastfuncstuff.laminar.experiment import depth_inclusion_probabilities

    targets = layer_model_targets(3)
    F = [-1e6] * 8
    F[targets.index((1, 1, 1))] = 0.0
    inc = depth_inclusion_probabilities(F).numpy()
    assert np.allclose(inc, [1.0, 1.0, 1.0], atol=1e-9)

    F = [-1e6] * 8
    F[0] = 0.0  # null
    inc = depth_inclusion_probabilities(F).numpy()
    assert np.allclose(inc, [0.0, 0.0, 0.0], atol=1e-9)


def test_model_probabilities_do_not_overflow_on_a_wide_spread():
    """Bug of record: the reference shifts by the minimum, which overflows.

    Subtracting the minimum leaves the best model's exponent as large as the
    entire spread of F, so a spread past ~709 nats gives inf/inf = nan for every
    model -- a silent, total loss of the result. The example ROI spans 232 nats
    and is safe; a stronger effect or a wider model space would not be.
    """
    from fastfuncstuff.laminar.experiment import posterior_model_probabilities as p

    wide = [-1e6] * 8
    wide[3] = 0.0
    probs = p(wide).numpy()
    assert np.isfinite(probs).all(), "wide free-energy spread overflowed"
    assert probs[3] == pytest.approx(1.0)
    assert probs.sum() == pytest.approx(1.0)

    # Unchanged where the reference does not overflow.
    ordinary = [-352.98, -276.81, -237.18, -140.48, -216.50, -125.72, -131.80, -120.78]
    f = np.asarray(ordinary)
    naive = np.exp(f - f.min()) / np.exp(f - f.min()).sum()
    assert np.abs(p(ordinary).numpy() - naive).max() < 1e-12


# --------------------------------------------------------------------------
# Lockstep batching
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_lockstep_is_bit_identical_to_serial():
    """Sharing the forward call must not perturb a single fit.

    The whole design rests on this: each fit runs its own unchanged Variational
    Laplace scheme and only the integration is shared, so anything other than an
    exact match means the coroutine refactor changed the arithmetic.
    """
    from fastfuncstuff.laminar.inversion import (
        variational_laplace,
        variational_laplace_lockstep,
    )

    o = _oracle()
    block = o["per_k"][1]
    K, ns = int(block["K"]), int(block["ns"])
    spec = _spec(o, K)

    def t(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)

    y = t(block["y"]).reshape(ns, K)
    u = t(block["u"]).reshape(-1, 2)
    rows = t(block["mask_rows"]).reshape(-1).to(torch.bool)
    kernel = t(o["kernel"]).reshape(-1)

    targets = layer_model_targets(3)[:4]  # null and the three singles
    priors = [faes_priors(spec, tg) for tg in targets]

    serial = [variational_laplace(spec, pr, u, y, rows=rows, kernel=kernel) for pr in priors]
    lock = variational_laplace_lockstep(
        spec, priors, u, [y] * len(priors), rows=rows, kernel=kernel
    )

    assert len(lock) == len(serial)
    for a, c, tg in zip(serial, lock, targets, strict=True):
        assert float(a.F) == float(c.F), f"{tg}: F differs"
        assert a.n_iter == c.n_iter, f"{tg}: iteration count differs"
        assert torch.equal(vec(a.Ep), vec(c.Ep)), f"{tg}: posterior means differ"


def test_lockstep_validates_its_inputs():
    from fastfuncstuff.laminar.inversion import variational_laplace_lockstep

    spec = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
    pr = faes_priors(spec, (1, 0, 0))
    y = torch.zeros(10, 6, dtype=torch.float64)
    u = torch.zeros(100, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="priors against"):
        variational_laplace_lockstep(spec, [pr, pr], u, [y])
    with pytest.raises(ValueError, match="scans, expected"):
        variational_laplace_lockstep(spec, [pr, pr], u, [y, torch.zeros(9, 6, dtype=torch.float64)])
