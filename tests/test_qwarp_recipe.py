"""ffs_qwarp -qwarp_recipe: a named bundle of settings that explicit flags still override."""

from fastfuncstuff.cli.qwarp import _recipe_value, parse_args
from fastfuncstuff.processing.warp import QWARP_RECIPES, QwarpConfig

BASE = ["-base", "b.nii.gz", "-source", "s.nii.gz", "-prefix", "out"]
# CLI dest -> QwarpConfig field, for every field a recipe can set
DESTS = {"patch_sweep": "patch_sweep", "gn_stop": "gn_stop_rule", "optimizer": "optimizer",
         "lev0": "lev0_schedule", "ball": "ball_constraint", "incor_afni": "incor_afni",
         "weight_afni": "weight_afni", "background_band": "background_band"}  # fmt: skip


def _resolved(args):
    return {field: _recipe_value(args, dest, field) for dest, field in DESTS.items()}


def test_every_recipe_field_has_a_flag():
    assert {f for r in QWARP_RECIPES.values() for f in r} <= set(DESTS.values())


def test_default_recipe_is_qwarpconfig():
    cfg = QwarpConfig()
    assert _resolved(parse_args(BASE)) == {f: getattr(cfg, f) for f in DESTS.values()}


def test_afni_recipe_sets_its_fields_and_flags_win():
    got = _resolved(parse_args([*BASE, "-qwarp_recipe", "afni"]))
    assert got == {
        f: QWARP_RECIPES["afni"].get(f, getattr(QwarpConfig(), f)) for f in DESTS.values()
    }
    assert got["optimizer"] == "hybrid" and got["patch_sweep"] == "afni"
    over = _resolved(parse_args([*BASE, "-qwarp_recipe", "afni", "-optimizer", "gn",
                                 "-background_band", "15"]))  # fmt: skip
    assert over["optimizer"] == "gn" and over["background_band"] == 15
    assert over["patch_sweep"] == "afni"  # the rest of the recipe stays


def test_batch_job_line_recipe_applies_over_top_level_defaults():
    """Batch mode re-parses each job line on top of the top-level namespace; a recipe
    on the job line must still take effect for flags the top level left unset."""
    top = parse_args(BASE)
    job = parse_args([*BASE, "-qwarp_recipe", "afni"], top)
    assert _recipe_value(job, "optimizer", "optimizer") == "hybrid"
