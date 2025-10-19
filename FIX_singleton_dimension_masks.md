# Fix: Handle Singleton Dimensions in AFNI Masks

## Issue
AFNI masks sometimes have singleton dimensions in the 4th dimension:
- Expected: `(64, 64, 35)` - 3D mask
- Actual: `(64, 64, 35, 1)` - 3D mask with singleton 4th dimension

This is technically a 4D array but represents a 3D mask. The old code rejected these files.

## Root Cause
AFNI's 3dcalc, 3dAutomask, and other tools sometimes create masks with shape `(nx, ny, nz, 1)` instead of `(nx, ny, nz)`. This is valid in AFNI but appeared as "4D" to our loader.

## Solution
Added `np.squeeze(data)` to automatically remove singleton dimensions:

```python
# Before
data = img.get_fdata(dtype=np.float32)
if data.ndim != 3:
    raise ValueError(...)  # ← Rejected (64, 64, 35, 1)

# After  
data = img.get_fdata(dtype=np.float32)
data = np.squeeze(data)  # ← Removes singleton dims
if data.ndim != 3:
    raise ValueError(...)  # ← Now accepts (64, 64, 35, 1) → (64, 64, 35)
```

## What `np.squeeze()` Does
- `(64, 64, 35, 1)` → `(64, 64, 35)` ✓
- `(64, 64, 35)` → `(64, 64, 35)` ✓ (no change)
- `(64, 64, 35, 2)` → `(64, 64, 35, 2)` (not squeezed, will fail check) ✓
- `(1, 64, 64, 35)` → `(64, 64, 35)` ✓ (leading singleton removed too)

## Updated Error Message
More helpful error message now mentions squeezing:
```
"Mask file must be 3D after squeezing singleton dimensions 
(received shape {shape} after squeeze)"
```

## Benefits
1. ✅ Handles AFNI masks with singleton dimensions
2. ✅ Works with both NIfTI and AFNI BRIK formats
3. ✅ Backwards compatible (3D masks still work)
4. ✅ Fails appropriately for true 4D files (e.g., time series)

## Testing
Test cases now handled:
- `mask.nii.gz` with shape `(64, 64, 35)` ✓
- `mask+tlrc.HEAD` with shape `(64, 64, 35, 1)` ✓
- `mask+tlrc.HEAD` with shape `(1, 64, 64, 35, 1)` ✓
- `timeseries.nii.gz` with shape `(64, 64, 35, 100)` ✗ (fails appropriately)

## Example Usage
```python
# Now works with AFNI masks that have singleton dimensions!
mask = ffs.load_afni_mask('mask+tlrc.HEAD', threshold=0.5)
# Even if file has shape (64, 64, 35, 1), returns (64, 64, 35)
```

## Related AFNI Commands
These AFNI commands commonly create masks with singleton dimensions:
```bash
3dcalc -a func+tlrc -expr 'step(a)' -prefix mask+tlrc
3dAutomask -prefix mask+tlrc func+tlrc
3dTstat -mean -prefix mean+tlrc func+tlrc
```

All now work perfectly with our loader! ✨
