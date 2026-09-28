"""ffs_denoise component-count defaults: fixed for PCA, estimated for ICA/dictionary."""

from argparse import Namespace

import pytest

from fastfuncstuff.cli.denoise import _resolve_component_count


def _args(**kw):
    base = dict(
        noise="pca",
        max_comps=None,
        auto_component_caps=False,
        no_auto_component_caps=False,
        auto_component_estimate_max=None,
    )
    base.update(kw)
    return Namespace(**base)


def test_pca_keeps_a_fixed_ceiling():
    a = _args(noise="pca")
    _resolve_component_count(a)
    assert a.max_comps == 20 and not a.auto_component_caps


@pytest.mark.parametrize("method", ["ica", "dictionary"])
def test_non_nested_methods_default_to_the_estimate(method):
    a = _args(noise=method)
    _resolve_component_count(a)
    assert a.auto_component_caps
    # the sweep ceiling follows the estimator's, not a hardcoded 20
    assert a.max_comps == a.auto_component_estimate_max == 40


def test_explicit_max_comps_is_respected_under_auto():
    a = _args(noise="ica", max_comps=12)
    _resolve_component_count(a)
    assert a.auto_component_caps and a.max_comps == 12


def test_opt_out_restores_the_fixed_count():
    a = _args(noise="dictionary", no_auto_component_caps=True)
    _resolve_component_count(a)
    assert not a.auto_component_caps and a.max_comps == 20


def test_contradictory_flags_exit():
    with pytest.raises(SystemExit):
        _resolve_component_count(
            _args(noise="ica", auto_component_caps=True, no_auto_component_caps=True)
        )
