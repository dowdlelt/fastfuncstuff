"""Every design-building call must use the RESOLVED model durations.

``resolve_hrf_library_spec`` returns zeros for a duration-convolved library:
the curve already contains the boxcar, so the design has to be built from
impulses.  The resolver was correct and the callers were not -- several kept
passing the physical ``durations`` into design construction, and because
``build_task_design`` prefers the event list and re-applies ``stim_durations``
itself, the duration went in twice.  Nothing about the resulting regressor
looks wrong.

This is a source-level check because the failure lives in the CLI-to-library
handoff, not in either side of it, and running five CLIs end-to-end to catch a
keyword argument is not a trade worth making.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[1] / "fastfuncstuff" / "cli"

# Calls that build (or parametrise) a task design from event timing.
DESIGN_BUILDERS = {
    "create_onset_matrix_microtime",
    "build_task_design_from_args",
    "create_single_trial_design",
    "_fit_voxelwise_hrf_single_trial",
}

DURATION_KWARGS = {"durations", "stim_durations"}

# Names that hold the resolved value.  Anything else -- above all the physical
# ``durations`` -- is the bug this test exists for.
RESOLVED_NAMES = {"model_durations", "onset_durations"}

TOOLS = ["denoise", "ridge", "hrfopt", "reml"]


# A call that deliberately builds with the canonical HRF (no library, so no
# boxcar is baked in) marks the line with this and says why.
EXEMPT_MARKER = "# canonical-hrf:"


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


@pytest.mark.parametrize("tool", TOOLS)
def test_design_builders_get_resolved_durations(tool):
    source = (CLI_DIR / f"{tool}.py").read_text()
    lines = source.split("\n")
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in DESIGN_BUILDERS:
            continue
        for kw in node.keywords:
            if kw.arg in DURATION_KWARGS and isinstance(kw.value, ast.Name):
                exempt = EXEMPT_MARKER in "\n".join(
                    lines[max(0, kw.value.lineno - 6) : kw.value.lineno]
                )
                if kw.value.id not in RESOLVED_NAMES and not exempt:
                    offenders.append(f"{tool}.py:{kw.value.lineno} {kw.arg}={kw.value.id}")

    assert not offenders, (
        "design built from unresolved durations (a duration-convolved HRF "
        "library would be convolved with the boxcar twice): " + ", ".join(offenders)
    )
