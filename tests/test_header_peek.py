"""Header reads without nibabel's full route: the byte peek and the catalog summary.

Both are speed paths with a slower reference beside them, so every test here
pins the fast answer to the reference on the same file. That's what stops a
wrong TR or a lost label from slipping into the viewer's list.
"""

import dataclasses
import gzip
import itertools
import struct

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.io import dsetinfo
from fastfuncstuff.io.headers import _read_leading_bytes, peek_nifti_summary
from fastfuncstuff.viewer import catalog

zstd = pytest.importorskip("compression.zstd")


def _skippable(payload: bytes) -> bytes:
    return struct.pack("<II", 0x184D2A50, len(payload)) + payload


@pytest.fixture
def payload():
    return np.random.default_rng(0).integers(0, 40, 600_000, dtype=np.uint8).tobytes()


class TestLeadingBytes:
    @pytest.mark.parametrize("n", [10, 544, 1 << 16, 250_000, 10_000_000])
    def test_zstd_frames_as_pzstd_writes_them(self, tmp_path, payload, n):
        """pzstd opens with a skippable frame, then writes one frame per job."""
        path = tmp_path / "x.nii.zst"
        cuts = [0, 100_000, 350_000, len(payload)]
        frames = [zstd.compress(payload[a:b]) for a, b in itertools.pairwise(cuts)]
        path.write_bytes(_skippable(b"\x00" * 4) + b"".join(frames))
        assert _read_leading_bytes(path, n) == payload[:n]

    @pytest.mark.parametrize("n", [10, 544, 1 << 16, 250_000, 10_000_000])
    def test_gzip_members_and_padding(self, tmp_path, payload, n):
        path = tmp_path / "x.nii.gz"
        half = len(payload) // 2
        path.write_bytes(
            gzip.compress(payload[:half]) + b"\x00" * 7 + gzip.compress(payload[half:])
        )
        assert _read_leading_bytes(path, n) == payload[:n]
        with gzip.open(path) as fh:
            assert fh.read(n) == payload[:n]

    @pytest.mark.parametrize("suffix", [".nii.zst", ".nii.gz"])
    def test_a_file_cut_inside_a_frame_raises(self, tmp_path, payload, suffix):
        """read_volume relies on this: a short read must not pass for voxels."""
        packed = zstd.compress(payload) if suffix == ".nii.zst" else gzip.compress(payload)
        path = tmp_path / f"x{suffix}"
        path.write_bytes(packed[: len(packed) // 2])
        with pytest.raises(EOFError):
            _read_leading_bytes(path, len(payload))


def _write(img, path):
    plain = path.with_name(path.name.removesuffix(".gz").removesuffix(".zst"))
    nib.save(img, str(plain))
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(plain.read_bytes()))
    elif path.name.endswith(".zst"):
        path.write_bytes(zstd.compress(plain.read_bytes()))
    return path


def _image(shape, *, t_unit="sec", tr=1.5, labels=None, endian="<"):
    hdr = nib.Nifti1Header(endianness=endian)
    img = nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4), header=hdr)
    img.header.set_xyzt_units(xyz="mm", t=t_unit)
    if len(shape) > 3:
        img.header["pixdim"][4] = tr
    if labels:
        from fastfuncstuff.io.afni import _set_afni_brick_labels

        _set_afni_brick_labels(img.header, labels)
    return img


CASES = {
    "series": dict(shape=(4, 5, 6, 7)),
    "msec": dict(shape=(4, 5, 6, 3), t_unit="msec", tr=2000.0),
    "usec": dict(shape=(4, 5, 6, 3), t_unit="usec", tr=800_000.0),
    "unset_unit": dict(shape=(4, 5, 6, 3), t_unit="unknown", tr=0.9),
    "no_tr": dict(shape=(4, 5, 6, 3), tr=0.0),
    "volume": dict(shape=(4, 5, 6)),
    "bucket_dim5": dict(shape=(4, 5, 6, 1, 3), labels=["a#0_Coef", "a#0_Tstat", "b"]),
    "labelled": dict(shape=(4, 5, 6, 2), labels=["first", "second"]),
    "big_endian": dict(shape=(4, 5, 6, 3), endian=">", labels=["x", "y", "z"]),
}


class TestSummaryAgreesWithReadInfo:
    @pytest.mark.parametrize("suffix", [".nii", ".nii.gz", ".nii.zst"])
    @pytest.mark.parametrize("case", list(CASES))
    def test_fields(self, tmp_path, case, suffix):
        img = _image(**CASES[case])
        path = _write(img, tmp_path / f"{case}{suffix}")
        if case == "big_endian":
            assert _read_leading_bytes(path, 4) == struct.pack(">i", 348)
        summary = peek_nifti_summary(path)
        info = dsetinfo.read_info(path)
        assert summary is not None
        assert summary.shape == info.shape[:3]
        assert summary.n_volumes == info.n_volumes
        assert summary.tr == float(info.tr)
        assert list(summary.labels) == info.labels

    def test_catalog_entries_match_the_full_route(self, tmp_path, monkeypatch):
        for case, kw in CASES.items():
            _write(_image(**kw), tmp_path / f"{case}_mask.nii.gz")
            _write(_image(**kw), tmp_path / f"{case}.nii.zst")
        paths = catalog.discover(tmp_path)
        fast = list(catalog.describe(paths))
        monkeypatch.setattr(catalog, "peek_nifti_summary", lambda p: None)
        assert fast == list(catalog.describe(paths))
        assert len(fast) == len(paths)


class TestSummaryDeclines:
    """None hands the file to read_info, which is always the safe answer."""

    def test_nifti2(self, tmp_path):
        path = tmp_path / "two.nii"
        nib.save(nib.Nifti2Image(np.zeros((3, 3, 3), np.float32), np.eye(4)), str(path))
        assert peek_nifti_summary(path) is None

    def test_selector_in_the_name(self, tmp_path):
        path = _write(_image((4, 5, 6, 7)), tmp_path / "run.nii")
        sel = path.with_name("run.nii[0..2]")
        path.rename(sel)
        assert peek_nifti_summary(sel) is None

    def test_extension_past_the_first_read(self, tmp_path):
        labels = [f"cond{i:03d}_" + "x" * 300 for i in range(250)]
        path = _write(_image((4, 4, 4, len(labels)), labels=labels), tmp_path / "b.nii.zst")
        assert peek_nifti_summary(path) is None
        entry = next(catalog.describe([path]))
        assert list(entry.labels) == labels

    def test_truncated_and_garbage(self, tmp_path):
        short = tmp_path / "short.nii"
        short.write_bytes(b"\x5c\x01\x00\x00" + b"\x00" * 100)
        junk = tmp_path / "junk.nii.gz"
        junk.write_bytes(b"not gzip at all" * 50)
        assert peek_nifti_summary(short) is None
        assert peek_nifti_summary(junk) is None
        assert list(catalog.describe([short, junk])) == []


def test_gifti_info_ignores_its_payloads(tmp_path, monkeypatch):
    """Emptying the <Data> bodies must not change a single field of the description."""
    from fastfuncstuff.io.gifti import save_gifti_data

    path = tmp_path / "run.lh.func.gii"
    save_gifti_data(
        path, np.random.default_rng(1).normal(size=(300, 5)), meta={"TR_seconds": "1.5"}
    )
    fast = dsetinfo.read_info(path)
    monkeypatch.setattr(dsetinfo, "_without_gifti_payloads", lambda raw: raw)
    slow = dsetinfo.read_info(path)
    a, b = dataclasses.asdict(fast), dataclasses.asdict(slow)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_equal(a[key], b[key], err_msg=key)
    assert fast.file_bytes == path.stat().st_size
