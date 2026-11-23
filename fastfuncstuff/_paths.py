import os
from pathlib import Path

_BENCHMARK_DATA_DIR: Path | None = None


def get_benchmark_data_dir() -> Path:
    global _BENCHMARK_DATA_DIR
    if _BENCHMARK_DATA_DIR is not None:
        return _BENCHMARK_DATA_DIR
    env = os.environ.get("FFS_BENCHMARK_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "test_data"


def set_benchmark_data_dir(path: Path | str) -> None:
    global _BENCHMARK_DATA_DIR
    _BENCHMARK_DATA_DIR = Path(path).resolve()


def get_bench_root() -> Path | None:
    """Root of the benchmark worklist (worklist.sqlite, cache/, units/).

    ``FFS_BENCH_ROOT`` if set, else ``bench/`` beside the benchmark data directory, so one
    variable moves both. Never defaults into the repository (Dropbox-synced): with neither
    variable set there is no root and the caller must ask for one.
    """
    env = os.environ.get("FFS_BENCH_ROOT")
    if env:
        return Path(env)
    data = os.environ.get("FFS_BENCHMARK_DATA_DIR")
    if data:
        return Path(data).parent / "bench"
    return None
