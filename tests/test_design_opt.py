"""Quick test to verify design optimization imports and basic functionality"""

import sys

import numpy as np

print("=" * 80)
print("Testing Design Optimization Module")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    import fastfuncsim as ffs
    print("   ✓ fastfuncsim imported")
except ImportError as e:
    print(f"   ✗ Failed to import fastfuncsim: {e}")
    sys.exit(1)

try:
    from fastfuncsim.design_optimization import (
        ISIConstraints,
        create_onset_matrix,
        generate_event_sequence,
        generate_isi_sequence,
    )
    print("   ✓ design_optimization functions imported")
except ImportError as e:
    print(f"   ✗ Failed to import design_optimization: {e}")
    sys.exit(1)

try:
    from fastfuncsim.metrics_empirical import (
        compute_detection_power_empirical,
        estimate_ar1_coefficient,
    )
    print("   ✓ metrics_empirical functions imported")
except ImportError as e:
    print(f"   ✗ Failed to import metrics_empirical: {e}")
    sys.exit(1)

# Test basic functionality
print("\n2. Testing basic functionality...")

# Test event sequence generation
print("   Testing event sequence generation...")
event_seq = generate_event_sequence(
    n_trials_per_condition=10,
    n_conditions=2,
    ordering='alternating',
    seed=42
)
print(f"   ✓ Generated event sequence: {event_seq[:10]}...")

# Test ISI generation
print("   Testing ISI generation...")
isi_constraints = ISIConstraints(min_isi=2.0, max_isi=8.0, mean_isi=4.0, tr=1.0)
isis = generate_isi_sequence(
    n_events=20,
    isi_constraints=isi_constraints,
    distribution='exponential',
    seed=42
)
print(f"   ✓ Generated ISIs: mean={isis.mean():.2f}, std={isis.std():.2f}")

# Test onset matrix creation
print("   Testing onset matrix creation...")
onsets = create_onset_matrix(
    event_sequence=event_seq,
    isis=isis,
    duration=100.0,
    tr=1.0,
    n_conditions=2
)
print(f"   ✓ Created onset matrix: shape={onsets.shape}")

# Test AR(1) estimation
print("   Testing AR(1) coefficient estimation...")
residuals = np.random.randn(100)
# Add some autocorrelation
for i in range(1, len(residuals)):
    residuals[i] = 0.3 * residuals[i-1] + 0.7 * residuals[i]
rho = estimate_ar1_coefficient(residuals)
print(f"   ✓ Estimated AR(1) coefficient: {rho:.3f}")

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nYou can now run example_design_optimization.py for a full demo.")
