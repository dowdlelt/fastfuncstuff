from __future__ import annotations

import torch

import fastfuncstuff._compile as compile_policy


def test_compile_after_eager_time_avoids_one_shot_compile(monkeypatch):
    compile_calls = []

    def fake_safe_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    clock = iter((0.0, 0.2))
    monkeypatch.setattr(compile_policy, "safe_compile", fake_safe_compile)
    monkeypatch.setattr(compile_policy.time, "perf_counter", lambda: next(clock))

    wrapped = compile_policy.compile_after_eager_time(lambda x: x + 1, min_eager_seconds=1.0)
    result = wrapped(torch.tensor(2.0))

    assert result.item() == 3.0
    assert compile_calls == []


def test_compile_after_eager_time_reuses_compile_after_break_even(monkeypatch):
    compile_calls = []
    compiled_calls = []

    def kernel(x):
        return x + 1

    def fake_safe_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def compiled(x):
            compiled_calls.append(True)
            return fn(x)

        return compiled

    clock = iter((0.0, 0.3, 0.3, 0.6))
    monkeypatch.setattr(compile_policy, "safe_compile", fake_safe_compile)
    monkeypatch.setattr(compile_policy.time, "perf_counter", lambda: next(clock))

    wrapped = compile_policy.compile_after_eager_time(
        kernel,
        min_eager_seconds=0.5,
        dynamic=True,
    )
    x = torch.tensor(2.0)

    assert wrapped(x).item() == 3.0
    assert wrapped(x).item() == 3.0
    assert compile_calls == []

    assert wrapped(x).item() == 3.0
    assert wrapped(x).item() == 3.0
    assert compile_calls == [{"dynamic": True}]
    assert len(compiled_calls) == 2
