"""The MNI_T1a presets: what -type actually sets on each engine.

A preset is a measurement that has been promoted to a default, so the thing
worth testing is that the promotion survives -- that the number in the registry
is the number the engine ends up with, in the engine's own units.
"""

import pytest

from fastfuncstuff.processing.tunespec import (
    BACKENDS,
    PRESETS,
    RECIPES,
    describe_presets,
    preferred_backend,
    preset_config_for_cli,
)

MNI_T1A_BACKENDS = ("optiwarp_hs", "optiwarp_gradient", "formwarp", "qwarp")


def test_every_mni_t1a_preset_names_a_real_backend_and_real_params():
    assert "MNI_T1a" in RECIPES
    for backend in MNI_T1A_BACKENDS:
        preset = PRESETS[("MNI_T1a", backend)]
        assert preset.dated, f"{backend} preset has no date"
        assert preset.provenance
        spec = BACKENDS[backend]
        for key in preset.config:
            spec.param(key)  # raises if the knob does not exist


def test_millimetre_values_convert_with_the_grid():
    """Stored in mm so a preset transfers between resolutions -- the whole point."""
    at_1mm = preset_config_for_cli("MNI_T1a", "optiwarp_hs", (1.0, 1.0, 1.0))
    at_07mm = preset_config_for_cli("MNI_T1a", "optiwarp_hs", (0.7, 0.7, 0.7))

    assert at_1mm["update_sigma"] == pytest.approx(1.0)
    assert at_07mm["update_sigma"] == pytest.approx(1.0 / 0.7)
    # Unitless knobs must NOT be rescaled by the voxel size.
    assert at_1mm["conv_window"] == at_07mm["conv_window"] == 10
    assert at_1mm["jac_floor"] == at_07mm["jac_floor"] == 0.0


def test_force_model_travels_with_the_preset():
    """-type has to be the whole answer, not most of it."""
    cfg = preset_config_for_cli("MNI_T1a", "optiwarp_hs", (1.0, 1.0, 1.0))
    assert cfg["force"] == "hs"
    assert preferred_backend("MNI_T1a", "optiwarp", "optiwarp_demons") == "optiwarp_hs"
    # The older recipe still means demons; a new letter must not move an old one.
    assert preferred_backend("MNI_T1", "optiwarp", "optiwarp_demons") == "optiwarp_demons"


def test_help_lists_one_flag_per_line_and_marks_the_default():
    """argparse wraps, so a settings list on one line comes out broken mid-value."""
    text = describe_presets("optiwarp")
    lines = [ln.strip() for ln in text.split("\n")]
    assert "MNI_T1a  (set 2026-09-10)  [default]" in lines
    assert "-jac_floor 0.0" in lines
    assert "-update_sigma 1.0 mm" in lines
    assert any(ln.startswith("MNI_T1a") and "[-force gradient]" in ln for ln in lines)
    # The family listing must cover every engine behind the one command.
    assert any("-force demons" == ln for ln in lines)


def test_qwarp_help_points_at_the_better_engine():
    text = describe_presets("qwarp")
    assert "MNI_T1a" in text
    assert "ffs_optiwarp" in text
