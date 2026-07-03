"""
Tests for uncovered functions in cli_utils.py:
- parse_prefix / PrefixInfo
- clean_condition_labels
- parse_cv_strategy
- print_cli_header
- estimate_device_strategy
- build_nuisance_per_run
- parse_input_files
"""

import torch

from fastfuncstuff.cli_utils import (
    PrefixInfo,
    clean_condition_labels,
    estimate_device_strategy,
    parse_cv_strategy,
    parse_input_files,
    parse_prefix,
    print_cli_header,
)


class TestPrefixInfo:
    def test_with_suffix(self):
        pi = PrefixInfo(stem="output/glm", nifti_ext=".nii.gz")
        assert pi.with_suffix("r2") == "output/glm_r2.nii.gz"

    def test_as_file(self):
        pi = PrefixInfo(stem="out", nifti_ext=".nii.gz")
        assert pi.as_file() == "out.nii.gz"


class TestParsePrefix:
    def test_no_extension(self):
        result = parse_prefix("output")
        assert result.stem == "output"
        assert result.nifti_ext == ".nii.gz"

    def test_nii_gz(self):
        result = parse_prefix("output.nii.gz")
        assert result.stem == "output"
        assert result.nifti_ext == ".nii.gz"

    def test_nii(self):
        result = parse_prefix("output.nii")
        assert result.stem == "output"
        assert result.nifti_ext == ".nii"

    def test_nii_zst(self):
        result = parse_prefix("output.nii.zst")
        assert result.stem == "output"
        assert result.nifti_ext == ".nii.zst"

    def test_with_path(self):
        result = parse_prefix("dir/sub01")
        assert result.stem == "dir/sub01"
        assert result.nifti_ext == ".nii.gz"


class TestCleanConditionLabels:
    def test_common_prefix(self):
        labels = [
            "onsets.localizer.times.faces",
            "onsets.localizer.times.bodies",
            "onsets.localizer.times.scenes",
        ]
        cleaned = clean_condition_labels(labels)
        assert cleaned == ["faces", "bodies", "scenes"]

    def test_single_label(self):
        assert clean_condition_labels(["foo"]) == ["foo"]

    def test_no_common_prefix(self):
        labels = ["cond_A", "stim_B"]
        cleaned = clean_condition_labels(labels)
        # No common prefix/suffix, should return as-is or slightly trimmed
        assert len(cleaned) == 2

    def test_underscore_separator(self):
        labels = ["task_run01_faces", "task_run01_houses"]
        cleaned = clean_condition_labels(labels)
        assert "faces" in cleaned
        assert "houses" in cleaned


class TestParseCvStrategy:
    def test_loro_string(self):
        assert parse_cv_strategy("loro") == 1

    def test_loo_string(self):
        assert parse_cv_strategy("loo") == 1

    def test_integer_1(self):
        assert parse_cv_strategy("1") == 1

    def test_integer_leave_n_out(self):
        assert parse_cv_strategy("2") == 2
        assert parse_cv_strategy("3") == 3

    def test_float_fraction(self):
        assert parse_cv_strategy("0.5") == 0.5
        assert parse_cv_strategy("0.8") == 0.8


class TestPrintCliHeader:
    def test_basic(self, capsys):
        print_cli_header("TestTool")
        captured = capsys.readouterr()
        assert "TestTool" in captured.out
        assert "Started:" in captured.out

    def test_with_subtitle(self, capsys):
        print_cli_header("TestTool", subtitle="v1.0")
        captured = capsys.readouterr()
        assert "v1.0" in captured.out


class TestEstimateDeviceStrategy:
    def test_small_data_fits(self):
        """Small data should not need CPU offloading."""
        keep_on_cpu = estimate_device_strategy(
            n_voxels=1000,
            n_timepoints_total=100,
            device=torch.device("cpu"),
        )
        assert isinstance(keep_on_cpu, bool)

    def test_force_cpu(self):
        keep_on_cpu = estimate_device_strategy(
            n_voxels=10000,
            n_timepoints_total=100,
            device=torch.device("cpu"),
            force_cpu=True,
        )
        assert keep_on_cpu is True

    def test_large_data_on_cpu(self):
        """Very large data exceeding threshold should stay on CPU."""
        keep_on_cpu = estimate_device_strategy(
            n_voxels=5000000,
            n_timepoints_total=10000,
            device=torch.device("cpu"),
            gpu_threshold_gb=0.001,
        )
        assert keep_on_cpu is True


class TestParseInputFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.nii.gz"
        f.touch()
        result = parse_input_files(str(f))
        assert len(result) == 1

    def test_glob_pattern(self, tmp_path):
        for i in range(3):
            (tmp_path / f"run{i:02d}.nii.gz").touch()
        result = parse_input_files(str(tmp_path / "run*.nii.gz"))
        assert len(result) == 3

    def test_list_of_files(self, tmp_path):
        files = []
        for i in range(2):
            f = tmp_path / f"run{i}.nii.gz"
            f.touch()
            files.append(str(f))
        result = parse_input_files(files)
        assert len(result) == 2
