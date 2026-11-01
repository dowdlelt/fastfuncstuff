"""Tripwire for the cast orders Metal gets wrong, and for what depends on them.

torch 2.14: naming a device and a 64-bit dtype in one call on an MPS tensor returns
**zeros** without raising, and casting to float64 before leaving Metal raises. Both
read as perfectly good code on the CUDA box where they are written, so they are
enforced here rather than remembered. See "MPS traps in torch 2.14" in the wiki.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import fastfuncstuff
from fastfuncstuff.cli_utils import parse_device_arg, setup_device
from fastfuncstuff.glm.arma import _ljung_box_batched
from fastfuncstuff.utils import MPS_DEVICE, get_device

_F64 = {"torch.float64", "torch.double"}
_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Metal")


def _violations(source: str) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords}
        args = [ast.unparse(a) for a in node.args]
        if node.func.attr == "to":
            combined = (kwargs.get("dtype") in _F64 and "device" in kwargs) or (
                len(args) == 2 and args[1] in _F64
            )
            if combined:
                out.append((node.lineno, "device and float64 in one .to()"))
        moving_off = node.func.attr == "cpu"
        receiver = node.func.value
        if moving_off and isinstance(receiver, ast.Call):
            inner = ast.unparse(receiver.func)
            casted = inner.endswith(".double") or (
                inner.endswith(".to") and [ast.unparse(a) for a in receiver.args] in ([f] for f in _F64)
            )
            if casted:
                out.append((node.lineno, "float64 cast before .cpu()"))
    return out


def test_detector_catches_each_form():
    """The scan below is only worth anything if these keep being caught."""
    for bad in (
        "x.to('cpu', torch.float64)",
        "x.to(device=d, dtype=torch.float64)",
        "x.to(dtype=torch.double, device=d)",
        "x.double().cpu()",
        "x.to(torch.float64).cpu()",
    ):
        assert _violations(bad), bad
    for good in ("x.to(device=d).to(torch.float64)", "x.cpu().double()", "x.to(d, torch.float32)"):
        assert not _violations(good), good


def test_package_has_no_unsafe_float64_casts():
    root = Path(fastfuncstuff.__file__).parent
    found = [
        f"{path.relative_to(root)}:{line}: {what}"
        for path in sorted(root.rglob("*.py"))
        for line, what in _violations(path.read_text())
    ]
    assert not found, "\n".join(found)


def test_every_device_source_hands_out_the_indexed_mps_device():
    """A tensor reports mps:0, and torch.device("mps") != torch.device("mps:0")."""
    assert MPS_DEVICE == torch.device("mps:0")
    assert parse_device_arg("mps")[0] == MPS_DEVICE


def test_float64_only_tool_runs_on_cpu_when_mps_is_requested(monkeypatch):
    monkeypatch.setattr(
        "fastfuncstuff.cli_utils.parse_device_arg", lambda spec: (MPS_DEVICE, None, None)
    )
    with pytest.warns(UserWarning, match="ffs_test_tool is float64-only"):
        assert setup_device("mps", float64_only="ffs_test_tool").type == "cpu"
    # Without the flag the request stands: Metal is only refused by the tool that needs to.
    monkeypatch.setattr("fastfuncstuff.utils.configure_torch_backends", lambda *a, **k: None)
    assert setup_device("mps") == MPS_DEVICE


@pytest.mark.gpu
@_mps
def test_get_device_compares_equal_to_its_own_tensors():
    assert torch.zeros(1, device=get_device("mps")).device == get_device("mps")


@pytest.mark.gpu
@_mps
def test_ljung_box_on_metal_matches_cpu():
    resid = torch.randn(50, 40, generator=torch.Generator().manual_seed(0))
    got = _ljung_box_batched(resid.to("mps"))
    assert got.dtype == torch.float64
    # Same float32 input and both reductions run on the CPU: identical.
    assert torch.equal(got, _ljung_box_batched(resid))
