#!/usr/bin/env python3
"""
Pre-flight checklist before running long experiments.
Catches issues before you waste time loading data!
"""

import sys
import subprocess
from pathlib import Path


def check_item(name, check_func):
    """Run a check and print status"""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name}")
            return False
    except Exception as e:
        print(f"⚠️  {name}: {e}")
        return False


def check_imports():
    """Check critical imports work"""
    try:
        import torch
        import numpy as np
        import nibabel
        from fastfuncsim import fit_glm_arma11, analyze_from_design_matrix

        return True
    except ImportError as e:
        print(f"   Missing: {e}")
        return False


def check_gpu():
    """Check GPU is available"""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Device: {device} ({mem:.1f} GB)")
            return True
        else:
            print("   CUDA not available")
            return False
    except:
        return False


def check_types():
    """Run critical type check"""
    result = subprocess.run(
        ["python", "check_types.py"], capture_output=True, text=True
    )
    return result.returncode == 0


def main():
    print("=" * 70)
    print("🚀 PRE-FLIGHT CHECKLIST")
    print("=" * 70)

    checks = [
        ("Critical imports", check_imports),
        ("GPU available", check_gpu),
        ("Type checking", check_types),
    ]

    results = [check_item(name, func) for name, func in checks]

    print("=" * 70)
    if all(results):
        print("✅ ALL SYSTEMS GO! Ready to run.")
        print("=" * 70)
        return 0
    else:
        print("❌ Fix issues above before running long experiments!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
