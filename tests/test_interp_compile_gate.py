"""The eager-vs-compiled gate on the resample gather.

Compiling ``_gather_contract`` costs a ~2s warmup that inductor's FX graph cache
does NOT amortize across processes (it caches codegen, not dynamo tracing), so a
tool run as N short CLI invocations pays it N times. These tests pin the two
things that keep that from happening: the gate spends eager time before it
commits, and a caller can declare work one-shot so it never counts at all.

``torch.compile`` is stubbed out throughout — we are testing when the switch
happens, not what inductor emits.
"""

from __future__ import annotations

import json

import pytest
import torch

from fastfuncstuff.processing import interp

DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _isolated_gate(monkeypatch, tmp_path):
    """Reset the module-global budget and point calibration at a temp file."""
    monkeypatch.setattr(interp, "_eager_seconds", {"cpu": 0.0, "cuda": 0.0})
    monkeypatch.setattr(interp, "_compiled_gather_contract", {})
    monkeypatch.setattr(interp, "_compile_cost_cache", {})
    monkeypatch.setattr(interp, "_compile_pending_measure", set())
    monkeypatch.setattr(interp, "_pending_cuda_timings", [])
    monkeypatch.setattr(interp, "_no_compile_depth", 0)
    monkeypatch.setattr(interp, "_compile_cost_path", lambda: tmp_path / "cost.json")
    monkeypatch.delenv("FFS_NWARP_NO_COMPILE", raising=False)

    def _fake_compile(fn, **kwargs):
        marked = lambda *a, **k: fn(*a, **k)
        marked._is_compiled = True  # type: ignore[attr-defined]
        return marked

    monkeypatch.setattr(torch, "compile", _fake_compile)
    yield


def _is_compiled(fn) -> bool:
    return getattr(fn, "_is_compiled", False)


def _resample(nx: int = 24) -> None:
    """One wsinc5 resample of an nx^3 volume, through the gather path."""
    src = torch.randn(nx, nx, nx)
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(nx, dtype=torch.float32) + 0.3 for _ in range(3)), indexing="ij"
    )
    interp._separable_resample_3d(src, xx, yy, zz, "wsinc5")


class TestCompileGate:
    def test_first_calls_stay_eager(self):
        """A one-shot apply must not pay a warmup it can never earn back."""
        assert not _is_compiled(interp._get_gather_contract(DEV))
        _resample()
        assert not _is_compiled(interp._get_gather_contract(DEV))

    def test_compiles_once_eager_time_covers_the_warmup(self):
        interp._compile_cost_cache[interp._compile_cost_key("cpu")] = 0.5
        interp._eager_seconds["cpu"] = 0.49
        assert not _is_compiled(interp._get_gather_contract(DEV))
        interp._eager_seconds["cpu"] = 0.51
        assert _is_compiled(interp._get_gather_contract(DEV))

    def test_resampling_accumulates_the_budget(self):
        before = interp._eager_seconds["cpu"]
        _resample()
        assert interp._eager_seconds["cpu"] > before

    def test_env_override_pins_eager(self, monkeypatch):
        monkeypatch.setenv("FFS_NWARP_NO_COMPILE", "1")
        interp._compile_cost_cache[interp._compile_cost_key("cpu")] = 0.0
        interp._eager_seconds["cpu"] = 100.0
        assert not _is_compiled(interp._get_gather_contract(DEV))


class TestOneShotScope:
    def test_scope_keeps_the_gather_eager(self):
        interp._compile_cost_cache[interp._compile_cost_key("cpu")] = 0.0
        interp._eager_seconds["cpu"] = 100.0  # would compile immediately otherwise
        with interp.no_gather_compile():
            assert not _is_compiled(interp._get_gather_contract(DEV))
        assert _is_compiled(interp._get_gather_contract(DEV))

    def test_scope_work_does_not_feed_the_budget(self):
        """One-shot work must not push a later loop over the line on its own."""
        with interp.no_gather_compile():
            _resample()
        assert interp._eager_seconds["cpu"] == 0.0

    def test_scope_nests_and_restores(self):
        with interp.no_gather_compile():
            with interp.no_gather_compile():
                pass
            assert interp._no_compile_depth == 1
        assert interp._no_compile_depth == 0


class TestCalibration:
    def test_measured_cost_survives_the_process(self, tmp_path):
        interp._record_compile_cost("cpu", 3.25)
        stored = json.loads((tmp_path / "cost.json").read_text())
        assert stored[interp._compile_cost_key("cpu")] == pytest.approx(3.25)
        interp._compile_cost_cache.clear()
        assert interp._measured_compile_cost("cpu") == pytest.approx(3.25)

    def test_bootstrap_prior_when_uncalibrated(self):
        assert interp._measured_compile_cost("cpu") == interp._COMPILE_COST_BOOTSTRAP_S

    def test_key_is_per_torch_build(self):
        """A torch upgrade recalibrates instead of inheriting a stale number."""
        assert torch.__version__ in interp._compile_cost_key("cpu")

    def test_unreadable_cache_falls_back_to_the_prior(self, tmp_path):
        (tmp_path / "cost.json").write_text("{not json")
        assert interp._measured_compile_cost("cpu") == interp._COMPILE_COST_BOOTSTRAP_S

    def test_warmup_is_measured_on_the_first_compiled_call(self):
        interp._compile_cost_cache[interp._compile_cost_key("cpu")] = 0.0
        interp._eager_seconds["cpu"] = 100.0
        assert _is_compiled(interp._get_gather_contract(DEV))
        assert "cpu" in interp._compile_pending_measure
        _resample()
        assert "cpu" not in interp._compile_pending_measure  # measured and recorded
        interp._compile_cost_cache.clear()
        assert interp._measured_compile_cost("cpu") > 0.0


class TestResamplePlanCache:
    def test_reuses_plan_only_inside_scope(self, monkeypatch):
        import fastfuncstuff.memory as memory

        calls = []

        def available(device, *, empty_cache):
            calls.append((device, empty_cache))
            return 1_000_000

        monkeypatch.setattr(memory, "get_available_memory", available)
        with interp.cache_resample_plans():
            first = interp._resample_chunk_size(100, 8, DEV)
            second = interp._resample_chunk_size(100, 8, DEV)
        third = interp._resample_chunk_size(100, 8, DEV)

        assert first == second == third
        assert len(calls) == 2
