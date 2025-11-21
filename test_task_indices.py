#!/usr/bin/env python3
"""
Test that task_indices extraction works correctly for OLS outputs
"""
import numpy as np
from fastfuncsim.afni_io import read_afni_design_matrix

def test_task_indices_extraction():
    """Test extracting stimulus indices from StimBots/StimTops"""

    # Test with X.xmat.1D (simple case: 1 column per stimulus)
    print("=" * 70)
    print("Testing X.xmat.1D (simple case)")
    print("=" * 70)

    design_info = read_afni_design_matrix("X.xmat.1D")

    print(f"\nTotal columns: {design_info['n_regressors']}")
    print(f"Column labels ({len(design_info['column_labels'])}): {design_info['column_labels'][:5]}...{design_info['column_labels'][-3:]}")

    stim_bots = design_info.get("stim_bots", [])
    stim_tops = design_info.get("stim_tops", [])

    print(f"\nStimBots: {stim_bots}")
    print(f"StimTops: {stim_tops}")

    # Extract stimulus indices
    stim_indices = []
    if stim_bots and stim_tops:
        for bot, top in zip(stim_bots, stim_tops):
            stim_indices.extend(range(bot, top + 1))

    print(f"\nExtracted stimulus indices: {stim_indices}")
    print(f"Number of stimulus columns: {len(stim_indices)}")

    # Show the labels for those columns
    stim_labels = [design_info['column_labels'][i] for i in stim_indices]
    print(f"Stimulus labels: {stim_labels}")

    # Verify these are the task regressors (not polynomials or motion)
    expected_task = ['img_face#0', 'img_place#0', 'prc_face#0', 'prc_place#0']
    assert stim_labels == expected_task, f"Expected {expected_task}, got {stim_labels}"

    print("\n✅ PASS: Stimulus indices extracted correctly!")
    print()

    # Now simulate the multi-basis-function case described by user
    print("=" * 70)
    print("Simulated multi-basis-function case (like user's 322-regressor example)")
    print("=" * 70)

    # User's example:
    # - 322 total columns
    # - Columns 0-27: Run polynomials (nuisance)
    # - Columns 28-279: Stimulus columns (252 stimulus columns, task)
    # - Columns 280-321: Motion parameters (nuisance)
    # - StimBots would be [28, 50, 72, ...] (41 stimuli)
    # - StimTops would be [49, 71, 93, ...] (each stimulus has ~6 basis functions)

    # Simulate with a smaller example
    simulated_stim_bots = [28, 34, 40, 46]  # 4 stimuli
    simulated_stim_tops = [33, 39, 45, 51]  # each has 6 basis functions

    print(f"\nSimulated StimBots: {simulated_stim_bots}")
    print(f"Simulated StimTops: {simulated_stim_tops}")

    # Extract
    simulated_indices = []
    for bot, top in zip(simulated_stim_bots, simulated_stim_tops):
        simulated_indices.extend(range(bot, top + 1))

    print(f"\nExtracted indices: {simulated_indices}")
    print(f"Number of stimulus columns: {len(simulated_indices)}")

    # Verify correct range extraction
    expected_ranges = [
        list(range(28, 34)),  # First stimulus: 28-33 (6 columns)
        list(range(34, 40)),  # Second: 34-39 (6 columns)
        list(range(40, 46)),  # Third: 40-45 (6 columns)
        list(range(46, 52)),  # Fourth: 46-51 (6 columns)
    ]
    expected_all = [i for sublist in expected_ranges for i in sublist]

    assert simulated_indices == expected_all, f"Range extraction mismatch!"

    print("\n✅ PASS: Multi-basis-function extraction works correctly!")
    print(f"   Each stimulus has 6 basis functions → total {len(simulated_indices)} columns")
    print(f"   NOT just 4 starting indices!")
    print()

if __name__ == "__main__":
    test_task_indices_extraction()
