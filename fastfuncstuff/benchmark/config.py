"""Benchmark configuration — YAML-driven dataset parameters.

Each dataset gets a YAML config defining subject, session, tasks, runs,
which stages to run, and stage-specific parameters (stim labels, contrasts, etc.).
BIDS directory structure + config labels determine file paths at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkConfig:
    """Dataset-specific benchmark parameters loaded from YAML."""

    dataset_id: str
    subject: str = "01"
    session: str = "01"
    tasks: dict[str, list[int]] = field(default_factory=dict)
    stages: list[str] | None = None  # None = all applicable
    stage_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for cache storage."""
        return {
            "dataset_id": self.dataset_id,
            "subject": self.subject,
            "session": self.session,
            "tasks": self.tasks,
            "stages": self.stages,
            "stage_params": self.stage_params,
        }


def load_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark config from a YAML file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing.
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    dataset_id = raw.get("dataset_id")
    if not dataset_id:
        raise ValueError("Config must specify 'dataset_id'")

    # Parse tasks: accept both {task: [runs]} and {task: {runs: [runs]}}
    tasks_raw = raw.get("tasks", {})
    tasks: dict[str, list[int]] = {}
    for task_name, task_val in tasks_raw.items():
        if isinstance(task_val, dict):
            tasks[task_name] = list(task_val.get("runs", []))
        elif isinstance(task_val, list):
            tasks[task_name] = list(task_val)
        else:
            raise ValueError(
                f"Task '{task_name}' must be a list of runs or a dict with 'runs' key, "
                f"got {type(task_val).__name__}"
            )

    return BenchmarkConfig(
        dataset_id=str(dataset_id),
        subject=str(raw.get("subject", "01")),
        session=str(raw.get("session", "01")),
        tasks=tasks,
        stages=raw.get("stages"),
        stage_params=raw.get("stage_params", {}),
    )


def default_config() -> BenchmarkConfig:
    """Return the ds005165 config matching the previously-hardcoded values."""
    return BenchmarkConfig(
        dataset_id="ds005165",
        subject="01",
        session="01",
        tasks={
            "localizer": [1, 2, 3, 4, 5],
            "rest": [1, 2, 3, 4, 5],
        },
        stages=[
            "moco", "slicetime", "crossalign", "align", "warp",
            "glm", "build_design", "ica", "ica_single", "automask",
            "glmsingle_prep", "glmsingle_matlab",
            "glmsingle_hrf", "glmsingle_denoise", "glmsingle_ridge",
            "glm_tent", "glm_im", "glm_im_reml",
        ],
        stage_params={
            "glm": {
                "primary_task": "localizer",
                "stim_labels": ["faces", "bodies", "objects", "scenes", "scrambled"],
                "glts": [
                    ["faces_vs_objects", "+1*faces -1*objects"],
                    ["faces_vs_scenes", "+1*faces -1*scenes"],
                    ["faces_vs_scrambled", "+1*faces -1*scrambled"],
                ],
                "hrf_model": "SPMG1(3)",
            },
            "ica": {
                "tasks": ["rest", "localizer"],
            },
            "crossalign": {
                "reference_task": "localizer",
                "reference_run": 1,
            },
        },
    )


def find_config(data_dir: Path) -> Path | None:
    """Search for a benchmark config file.

    Checks (in order):
    1. data_dir/benchmark_config.yaml
    2. Built-in configs matching the data directory name
    """
    # Check data directory itself
    local = data_dir / "benchmark_config.yaml"
    if local.exists():
        return local

    # Check built-in configs
    configs_dir = Path(__file__).parent / "configs"
    if configs_dir.exists():
        # Try to match dataset_id from directory name
        name = data_dir.name
        for suffix in ("-download", "_download", "-data"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        candidate = configs_dir / f"{name}.yaml"
        if candidate.exists():
            return candidate

    return None
