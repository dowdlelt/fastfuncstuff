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
