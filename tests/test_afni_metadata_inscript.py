"""Tests for in-script AFNI metadata writing (labels, stat params, arbitrary
attributes, provenance history) — the machinery that replaced the external
``3drefit`` write→relabel→compress loop.

These assert the encodings AFNI itself writes, so files round-trip through
3dinfo / the GUI. They would have caught the two bugs hit while building this:
using C backslash-escapes instead of NIML/XML entities, and losing provenance
history on plain-NIfTI outputs.
"""

from __future__ import annotations

import re

import nibabel as nib
import numpy as np

from fastfuncstuff.io.afni import (
    _niml_escape_string,
    read_brick_labels,
    read_brick_stataux,
    save_nifti,
    set_afni_atr,
    stat_type_to_stataux,
)


def _afni_ext(path) -> str:
    img = nib.load(str(path))
    return "".join(
        e.content.decode("utf-8", "replace") for e in img.header.extensions if e.get_code() == 4
    )


def _history(path) -> str | None:
    m = re.search(r'atr_name="HISTORY_NOTE"\s*>\s*\n\s*"([^"]*)"', _afni_ext(path), re.S)
    return m.group(1) if m else None


def test_stat_type_to_stataux_maps_prefixes():
    assert stat_type_to_stataux("fitt", (120,)) == (3, (120.0,))
    assert stat_type_to_stataux("fift", (2, 118)) == (4, (2.0, 118.0))
    assert stat_type_to_stataux("fizt", ()) == (5, ())
    # case-insensitive and tolerant of a leading '-'
    assert stat_type_to_stataux("-FITT", (10,)) == (3, (10.0,))


def test_labels_and_stataux_roundtrip(tmp_path):
    data = np.random.randn(5, 5, 5, 3).astype(np.float32)
    labels = ["Full_Fstat", "cond#0_Coef", "cond#0_Tstat"]
    stataux = {
        0: stat_type_to_stataux("fift", (2, 120)),
        2: stat_type_to_stataux("fitt", (120,)),
    }
    out = tmp_path / "bucket.nii.gz"
    save_nifti(data, out, affine=np.eye(4), brick_labels=labels, brick_stataux=stataux)

    img = nib.load(str(out))
    assert read_brick_labels(img) == labels
    assert read_brick_stataux(img) == {0: (4, (2.0, 120.0)), 2: (3, (120.0,))}


def test_niml_escape_matches_afni_entities():
    # AFNI's NI_quotize_string: &<>"' and CR/LF become XML entities.
    raw = "<a b=\"c\">\n&'x'"
    esc = _niml_escape_string(raw)
    assert esc == "&lt;a b=&quot;c&quot;&gt;&#x0a;&amp;&apos;x&apos;"
    # No raw special char survives that would break the enclosing XML.
    for ch in "<>\n":
        assert ch not in esc


def test_set_afni_atr_escapes_arbitrary_content(tmp_path):
    out = tmp_path / "x.nii.gz"
    save_nifti(np.zeros((3, 3, 3), np.float32), out, affine=np.eye(4))
    img = nib.load(str(out))
    payload = '<3dClustSim_NN1\n  thr="1-sided" >\n 1.0\n</3dClustSim_NN1>'
    set_afni_atr(img.header, "AFNI_CLUSTSIM_NN1_1sided", payload, ni_type="String")
    out2 = tmp_path / "x2.nii.gz"
    nib.save(img, str(out2))

    ext = _afni_ext(out2)
    m = re.search(r'atr_name="AFNI_CLUSTSIM_NN1_1sided"[^>]*>\s*"(.*?)"', ext, re.S)
    assert m is not None
    stored = m.group(1)
    # Stored as NIML entities, and no raw '<' that would corrupt the XML stream.
    assert "&lt;3dClustSim_NN1" in stored
    assert "&#x0a;" in stored
    assert "<3dClustSim_NN1" not in stored


def test_history_created_for_plain_nifti(tmp_path, monkeypatch):
    # Even a fresh plain-NIfTI output gains a HISTORY_NOTE (provenance for all
    # ffs_* saves), formatted like AFNI's tross_Append_History.
    monkeypatch.setattr("sys.argv", ["ffs_demo", "-input", "a<b>.nii"])
    out = tmp_path / "h.nii.gz"
    save_nifti(np.zeros((3, 3, 3), np.float32), out, affine=np.eye(4))
    hist = _history(out)
    assert hist is not None
    assert hist.endswith("ffs_demo -input a&lt;b&gt;.nii")
    assert re.match(r"\[.+@.+: .+\] ", hist)  # [user@host: ctime] prefix


def test_history_appends_on_resave(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ffs_first", "-x"])
    a = tmp_path / "a.nii.gz"
    save_nifti(np.zeros((3, 3, 3), np.float32), a, affine=np.eye(4))

    monkeypatch.setattr("sys.argv", ["ffs_second", "-y"])
    b = tmp_path / "b.nii.gz"
    img = nib.load(str(a))
    save_nifti(np.asarray(img.dataobj), b, header=img.header)

    hist = _history(b)
    assert hist is not None
    # Two entries separated by AFNI's encoded newline.
    assert "ffs_first -x" in hist
    assert "ffs_second -y" in hist
    assert "&#x0a;" in hist
