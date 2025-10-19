#!/usr/bin/env python3
"""
Quick type check for critical errors before running long experiments.

Usage:
    python check_types.py

Returns non-zero exit code if critical errors found.
"""

import subprocess
import sys


def run_pyright():
    """Run pyright and filter for critical errors"""

    # Run pyright on main library files
    result = subprocess.run(
        ["pyright", "fastfuncsim/*.py"], capture_output=True, text=True
    )

    # Look for critical patterns
    critical_patterns = [
        "ModuleNotFoundError",
        "ImportError",
        "cannot be resolved",
        "is not exported from module",
        "got an unexpected keyword argument",
    ]

    output = result.stdout + result.stderr
    lines = output.split("\n")

    critical_errors = []
    for line in lines:
        if "error:" in line.lower():
            for pattern in critical_patterns:
                if pattern.lower() in line.lower():
                    critical_errors.append(line)
                    break

    if critical_errors:
        print("🚨 CRITICAL TYPE ERRORS FOUND:")
        print("=" * 70)
        for error in critical_errors:
            print(error)
        print("=" * 70)
        print(
            f"\nFound {len(critical_errors)} critical errors that will break at runtime!"
        )
        return 1
    else:
        print("✅ No critical type errors found. Safe to run!")
        return 0


if __name__ == "__main__":
    sys.exit(run_pyright())
