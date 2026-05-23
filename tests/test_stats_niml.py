"""
Tests for stats/niml.py — 3dClustSim-style NIML table writer and
3drefit command generation for ffs_perm.

These outputs land in the AFNI cluster panel; a bug that drops a brick
label or mis-orders the table silently corrupts every reported cluster.
"""

from __future__ import annotations

import base64
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.stats.niml import (
    build_refit_commands,
    resolve_mask_idcode,
    write_clustsim_niml,
    write_mask_b64,
)


# ---------------------------------------------------------------------------
# write_clustsim_niml — file format pinning
# ---------------------------------------------------------------------------

def _make_table(npthr: int = 3, nathr: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 50.0, size=(npthr, nathr)).astype(np.float32)


class TestWriteClustsimNiml:
    def test_header_contains_all_required_attrs(self, tmp_path):
        out = tmp_path / "n1.niml"
        table = _make_table(3, 4)
        write_clustsim_niml(
            out, table,
            nn=1, sidedness="1-sided",
            commandline="ffs_perm -in x.nii.gz",
            nxyz=(64, 64, 32), dxyz=(2.5, 2.5, 3.0),
            pthr=(0.05, 0.02, 0.01),
            athr=(0.10, 0.05, 0.02, 0.01),
            n_perms=1000,
            mask_count=12345,
        )
        text = out.read_text()
        assert text.startswith("<3dClustSim_NN1")
        for needle in [
            'ni_type="4*float"',
            'ni_dimen="3"',
            'thresholding="1-sided"',
            'nxyz="64,64,32"',
            'dxyz="2.5000,2.5000,3.0000"',
            'fwhmxyz="0,0,0"',
            'iter="1000"',
            'mask_count="12345"',
            'commandline="ffs_perm -in x.nii.gz"',
        ]:
            assert needle in text, f"missing {needle!r} in NIML output"
        assert text.rstrip().endswith("</3dClustSim_NN1>")

    def test_pthr_athr_format_six_decimals(self, tmp_path):
        out = tmp_path / "n1.niml"
        write_clustsim_niml(
            out, _make_table(2, 2),
            nn=1, sidedness="2-sided",
            commandline="cmd",
            nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
            pthr=(0.05, 0.01), athr=(0.05, 0.01),
            n_perms=10,
        )
        text = out.read_text()
        # 6-decimal formatting for pthr/athr
        assert 'pthr="0.050000,0.010000"' in text
        assert 'athr="0.050000,0.010000"' in text

    def test_closing_bracket_position(self, tmp_path):
        """Without mask_count, the closing '>' goes on the last attribute line.
        With mask_count, it goes on the mask_count line."""
        out = tmp_path / "n1.niml"
        write_clustsim_niml(
            out, _make_table(1, 1),
            nn=1, sidedness="1-sided",
            commandline="cmd",
            nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
            pthr=(0.05,), athr=(0.05,),
            n_perms=1,
        )
        text = out.read_text()
        # Without mask_count, the closing '>' is appended to the athr line.
        lines = text.splitlines()
        athr_idx = next(i for i, ln in enumerate(lines) if "athr=" in ln)
        assert lines[athr_idx].rstrip().endswith(">"), lines[athr_idx]
        assert "mask_count" not in text

    def test_closing_bracket_with_mask_count(self, tmp_path):
        out = tmp_path / "n1.niml"
        write_clustsim_niml(
            out, _make_table(1, 1),
            nn=1, sidedness="1-sided",
            commandline="cmd",
            nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
            pthr=(0.05,), athr=(0.05,),
            n_perms=1, mask_count=42,
        )
        text = out.read_text()
        lines = text.splitlines()
        mc_line = next(ln for ln in lines if "mask_count=" in ln)
        assert mc_line.rstrip().endswith(">")
        # And the athr line must NOT carry the '>'
        athr_line = next(ln for ln in lines if "athr=" in ln)
        assert not athr_line.rstrip().endswith(">")

    def test_table_body_matches_input_values(self, tmp_path):
        out = tmp_path / "n1.niml"
        # Choose integer values so the _fmt path that emits int(v) fires.
        table = np.array([[1.0, 2.0], [3.0, 4.5]], dtype=np.float32)
        write_clustsim_niml(
            out, table,
            nn=2, sidedness="bi-sided",
            commandline="cmd",
            nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
            pthr=(0.05, 0.01), athr=(0.10, 0.05),
            n_perms=1,
        )
        text = out.read_text()
        # Integers without trailing .0
        assert re.search(r"^\s+1 2\s*$", text, flags=re.MULTILINE)
        # Non-integer renders with %.7g
        assert "4.5" in text

    def test_nn_tag_matches_argument(self, tmp_path):
        for nn in (1, 2, 3):
            out = tmp_path / f"n{nn}.niml"
            write_clustsim_niml(
                out, _make_table(1, 1),
                nn=nn, sidedness="1-sided",
                commandline="cmd",
                nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
                pthr=(0.05,), athr=(0.05,), n_perms=1,
            )
            text = out.read_text()
            assert text.startswith(f"<3dClustSim_NN{nn}")
            assert text.rstrip().endswith(f"</3dClustSim_NN{nn}>")

    def test_optional_mask_attrs_emitted_when_present(self, tmp_path):
        out = tmp_path / "n1.niml"
        write_clustsim_niml(
            out, _make_table(1, 1),
            nn=1, sidedness="1-sided",
            commandline="cmd",
            nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
            pthr=(0.05,), athr=(0.05,), n_perms=1,
            mask_idcode="AFN_ABCDEFGH",
            mask_name="brain_mask.nii.gz",
        )
        text = out.read_text()
        assert 'mask_dset_idcode="AFN_ABCDEFGH"' in text
        assert 'mask_dset_name="brain_mask.nii.gz"' in text

    def test_shape_mismatch_raises(self, tmp_path):
        out = tmp_path / "n1.niml"
        # Table is (3,4) but pthr/athr declare (2,3)
        with pytest.raises(ValueError, match="table shape"):
            write_clustsim_niml(
                out, _make_table(3, 4),
                nn=1, sidedness="1-sided", commandline="cmd",
                nxyz=(1, 1, 1), dxyz=(1.0, 1.0, 1.0),
                pthr=(0.05, 0.01), athr=(0.10, 0.05, 0.02), n_perms=1,
            )


# ---------------------------------------------------------------------------
# write_mask_b64 — round-trip AFNI's mask_to_b64string
# ---------------------------------------------------------------------------

def _decode_mask_b64(path: Path) -> tuple[np.ndarray, int]:
    """Invert write_mask_b64 to recover the bitmask and declared nvox."""
    text = path.read_text()
    # b64 itself may end in '=' padding, so use rpartition to find the marker.
    body, sep, tail = text.rpartition("===")
    assert sep == "===", "missing ===<nvox> trailer"
    nvox_declared = int(tail.strip())
    # Strip newlines/whitespace from base64 body
    b64 = "".join(body.split())
    compressed = base64.b64decode(b64)
    packed = zlib.decompress(compressed)
    flat = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    flat = flat[:nvox_declared]
    return flat.astype(bool), nvox_declared


class TestWriteMaskB64:
    def test_round_trip_recovers_mask(self, tmp_path):
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(5, 4, 3)).astype(bool)
        out = tmp_path / "mask.b64"
        n_in = write_mask_b64(out, mask)
        assert n_in == int(mask.sum())
        flat_back, nvox = _decode_mask_b64(out)
        assert nvox == mask.size
        # write_mask_b64 ravels in F-order, so compare against that
        np.testing.assert_array_equal(flat_back, mask.ravel(order="F"))

    def test_line_wrap_at_72_chars(self, tmp_path):
        mask = np.zeros((20, 20, 5), dtype=bool)
        mask[::2, ::2, :] = True
        out = tmp_path / "mask.b64"
        write_mask_b64(out, mask)
        text = out.read_text()
        body, _, _ = text.partition("===")
        for line in body.splitlines():
            assert len(line) <= 72

    def test_trailer_nvox_matches_total_voxel_count(self, tmp_path):
        mask = np.zeros((3, 4, 5), dtype=bool)
        mask[0, 0, 0] = True
        out = tmp_path / "mask.b64"
        write_mask_b64(out, mask)
        _, nvox = _decode_mask_b64(out)
        assert nvox == 3 * 4 * 5

    def test_returns_count_of_true_voxels(self, tmp_path):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[:3, :, :] = True
        out = tmp_path / "mask.b64"
        assert write_mask_b64(out, mask) == 300


# ---------------------------------------------------------------------------
# resolve_mask_idcode — deterministic fallback when 3dinfo absent
# ---------------------------------------------------------------------------

class TestResolveMaskIdcode:
    def test_none_returns_stable_fallback(self):
        a = resolve_mask_idcode(None)
        b = resolve_mask_idcode(None)
        assert a == b
        assert a.startswith("AFN_")
        assert len(a) == 4 + 22  # "AFN_" + 22 chars

    def test_nonexistent_path_falls_back_deterministically(self, tmp_path, monkeypatch):
        # Force 3dinfo to be absent so we exercise the hash path.
        monkeypatch.setattr("shutil.which", lambda name: None)
        p = tmp_path / "ghost.nii.gz"
        a = resolve_mask_idcode(p)
        b = resolve_mask_idcode(p)
        assert a == b
        assert a.startswith("AFN_")

    def test_different_paths_give_different_codes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        a = resolve_mask_idcode(tmp_path / "mask_a.nii.gz")
        b = resolve_mask_idcode(tmp_path / "mask_b.nii.gz")
        assert a != b


# ---------------------------------------------------------------------------
# build_refit_commands — dispatch logic
# ---------------------------------------------------------------------------

class TestBuildRefitCommands:
    def test_atrstring_cmd_built_for_niml_files(self, tmp_path):
        niml_files = {
            (1, "1sided"): tmp_path / "n1_1.niml",
            (2, "1sided"): tmp_path / "n2_1.niml",
        }
        for p in niml_files.values():
            p.write_text("dummy")
        cmds = build_refit_commands(
            stat_path=tmp_path / "stats.nii.gz",
            niml_files=niml_files,
            mask_b64_path=None,
        )
        assert len(cmds) == 1
        cmd = cmds[0]
        assert cmd[0] == "3drefit"
        # Each NIML entry produces one -atrstring pair, sorted by (nn, sided)
        assert cmd.count("-atrstring") == 2
        assert "AFNI_CLUSTSIM_NN1_1sided" in cmd
        assert "AFNI_CLUSTSIM_NN2_1sided" in cmd
        assert str(tmp_path / "stats.nii.gz") == cmd[-1]

    def test_atrstring_cmd_includes_mask_when_provided(self, tmp_path):
        cmds = build_refit_commands(
            stat_path=tmp_path / "stats.nii.gz",
            niml_files={(1, "1sided"): tmp_path / "n.niml"},
            mask_b64_path=tmp_path / "mask.b64",
        )
        assert any("AFNI_CLUSTSIM_MASK" in c for c in cmds[0])

    def test_labels_and_substatpar_in_separate_command(self, tmp_path):
        """3drefit refuses to combine -atrstring with -substatpar; the two
        must come back as separate commands."""
        cmds = build_refit_commands(
            stat_path=tmp_path / "stats.nii.gz",
            niml_files={(1, "1sided"): tmp_path / "n.niml"},
            mask_b64_path=None,
            brick_labels=["Beta#0", "Tstat#0"],
            stat_brick_indices=[1],
            dof=42,
        )
        assert len(cmds) == 2
        atrstring_cmd, refit_cmd = cmds
        assert "-atrstring" in atrstring_cmd
        assert "-atrstring" not in refit_cmd
        assert refit_cmd.count("-sublabel") == 2
        assert "-substatpar" in refit_cmd
        # fitt + dof present together
        substat_idx = refit_cmd.index("-substatpar")
        assert refit_cmd[substat_idx + 2] == "fitt"
        assert refit_cmd[substat_idx + 3] == "42"

    def test_no_commands_when_nothing_to_do(self, tmp_path):
        cmds = build_refit_commands(
            stat_path=tmp_path / "stats.nii.gz",
            niml_files={},
            mask_b64_path=None,
        )
        assert cmds == []

    def test_substatpar_requires_dof(self, tmp_path):
        """stat_brick_indices without dof is a no-op for that part of cmd2."""
        cmds = build_refit_commands(
            stat_path=tmp_path / "stats.nii.gz",
            niml_files={},
            mask_b64_path=None,
            stat_brick_indices=[1],
            dof=None,
        )
        # No commands at all: no NIML, no labels, no dof.
        assert cmds == []
