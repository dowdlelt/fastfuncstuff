"""The Triton compilation key is cached across processes, and invalidated."""

import json
import types

import pytest

from fastfuncstuff import triton_key as module


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path / "fastfuncstuff"


def _fake_triton(root):
    """A stand-in Triton installation: a package directory and a stub module."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "_C").mkdir(exist_ok=True)
    (root / "_C" / "libtriton.so").write_bytes(b"x" * 32)
    (root / "compiler.py").write_text("# stub\n")
    stub = types.ModuleType("triton")
    stub.__file__ = str(root / "__init__.py")
    stub.__version__ = "9.9.9"
    (root / "__init__.py").write_text("")
    return stub


class TestFingerprint:
    def test_source_edit_moves_the_fingerprint(self, tmp_path):
        triton = _fake_triton(tmp_path / "triton")
        before = module._fingerprint(triton)
        (tmp_path / "triton" / "compiler.py").write_text("# stub\n# edited\n")
        assert module._fingerprint(triton) != before

    def test_new_source_file_moves_the_fingerprint(self, tmp_path):
        triton = _fake_triton(tmp_path / "triton")
        before = module._fingerprint(triton)
        (tmp_path / "triton" / "extra.py").write_text("")
        assert module._fingerprint(triton) != before

    def test_rebuilt_library_moves_the_fingerprint(self, tmp_path):
        triton = _fake_triton(tmp_path / "triton")
        before = module._fingerprint(triton)
        (tmp_path / "triton" / "_C" / "libtriton.so").write_bytes(b"y" * 64)
        assert module._fingerprint(triton) != before


class TestStore:
    def test_round_trip(self, cache_dir):
        module._write("fp", "the-key")
        assert module._read("fp") == "the-key"

    def test_a_different_fingerprint_is_a_miss(self, cache_dir):
        """The whole safety argument: a stale key must never be served."""
        module._write("fp", "the-key")
        assert module._read("other") is None

    def test_corrupt_cache_is_a_miss_not_a_crash(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "triton_key.json").write_text("{not json")
        assert module._read("fp") is None

    def test_unwritable_cache_directory_is_survivable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "file"))
        (tmp_path / "file").write_text("not a directory")
        module._write("fp", "the-key")  # must not raise
        assert module._read("fp") is None


class TestInstall:
    def test_opt_out_leaves_triton_alone(self, monkeypatch):
        monkeypatch.setattr(module, "_installed", False)
        monkeypatch.setenv("FFS_NO_TRITON_KEY_CACHE", "1")
        assert module.install_triton_key_cache() is False

    def test_the_cached_key_matches_the_real_one(self, cache_dir, monkeypatch):
        triton_cache = pytest.importorskip("triton.runtime.cache")
        original = getattr(triton_cache, "triton_key", None)
        if original is None:
            pytest.skip("this Triton has no triton_key to cache")
        monkeypatch.setattr(module, "_installed", False)
        monkeypatch.setattr(module, "_patched", None)
        try:
            assert module.install_triton_key_cache() is True
            assert triton_cache.triton_key() == original.__wrapped__()
            assert json.loads((cache_dir / "triton_key.json").read_text())["key"]
        finally:
            triton_cache.triton_key = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
