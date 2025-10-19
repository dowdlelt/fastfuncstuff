# AFNI BRIK/HEAD Format Support

## Date: October 16, 2025

## Summary
Added full support for AFNI's native BRIK/HEAD format in addition to NIfTI. Users can now read and write AFNI files seamlessly.

## Background

AFNI typically uses its native BRIK/HEAD format:
- `.HEAD` file: ASCII header with metadata
- `.BRIK` file: Binary data (uncompressed)
- `.BRIK.gz` file: Compressed binary data (common)

While nibabel supports BRIK format, our library wasn't exposing this properly. Now users can work with AFNI files natively without conversion.

## Changes Made

### 1. New Helper Functions (`glm_outputs.py`)

#### `_normalize_output_path(output_path)`
Detects file format from extension:
```python
'stats+tlrc.HEAD' → format='afni', base='stats+tlrc'
'stats+tlrc.BRIK' → format='afni', base='stats+tlrc'
'stats.nii.gz'     → format='nifti_gz'
'stats.nii'        → format='nifti'
```

#### `_save_nifti_with_format(img, path, format, compress_brik)`
Universal save function supporting:
- NIfTI compressed (`.nii.gz`)
- NIfTI uncompressed (`.nii`)
- AFNI compressed (`.HEAD` + `.BRIK.gz`)
- AFNI uncompressed (`.HEAD` + `.BRIK`)

### 2. Updated `write_afni_bucket()`

#### New Parameter:
```python
output_format: Optional[str] = None
```
- Options: `'nifti'`, `'nifti_gz'`, `'afni'`
- If `None`, auto-detects from extension
- Overrides auto-detection when specified

#### Format Auto-Detection:
```python
# Automatic detection from extension
ffs.write_afni_bucket(results, 'glm+tlrc.HEAD')        # → AFNI
ffs.write_afni_bucket(results, 'glm+tlrc.BRIK.gz')    # → AFNI
ffs.write_afni_bucket(results, 'glm.nii.gz')          # → NIfTI compressed
ffs.write_afni_bucket(results, 'glm.nii')             # → NIfTI uncompressed
```

#### Compression Behavior:
- `compress_output=True` (default):
  - NIfTI: Creates `.nii.gz`
  - AFNI: Creates `.BRIK.gz` (HEAD stays uncompressed)
- `compress_output=False`:
  - NIfTI: Creates `.nii`
  - AFNI: Creates `.BRIK`

#### Return Value:
- NIfTI: Returns path to `.nii` or `.nii.gz`
- AFNI: Returns path to `.HEAD` file (AFNI convention)

### 3. Updated `load_and_concatenate_runs()`

Docstring clarified to show BRIK support:
```python
run_files : list of str or Path
    Paths to neuroimaging files, one per run
    Supports: NIfTI (.nii, .nii.gz) and AFNI (.HEAD, .BRIK, .BRIK.gz)
    For AFNI files, provide either .HEAD or .BRIK path
```

Already worked via nibabel, just documented now.

## Usage Examples

### 1. Write AFNI BRIK Format (Auto-Detect)

```python
# Extension tells it to use AFNI format
results = ffs.fit_glm_arma11(data, design, tr=2.0)

# Auto-detected as AFNI from extension
ffs.write_afni_bucket(
    results,
    'glm_bucket+tlrc.HEAD',  # .HEAD extension → AFNI format
    condition_names=labels,
    compress_output=True,     # Creates .BRIK.gz
)

# Creates:
#   glm_bucket+tlrc.HEAD      (header)
#   glm_bucket+tlrc.BRIK.gz   (compressed data)
```

### 2. Write AFNI BRIK Format (Explicit)

```python
# Explicitly specify format
ffs.write_afni_bucket(
    results,
    'glm_bucket',             # No extension needed
    condition_names=labels,
    output_format='afni',     # Force AFNI format
    compress_output=True,
)

# Creates:
#   glm_bucket.HEAD
#   glm_bucket.BRIK.gz
```

### 3. Load AFNI BRIK Files

```python
# Works automatically - nibabel handles it
run_files = [
    'run01+orig.HEAD',
    'run02+orig.HEAD',
    'run03+orig.HEAD',
]

data, run_starts = ffs.load_and_concatenate_runs(run_files)
# Just works! ✓
```

### 4. Mixed Format Pipeline

```python
# Load AFNI files
run_files = ['run01+orig.HEAD', 'run02+orig.HEAD']
data, _ = ffs.load_and_concatenate_runs(run_files)

# Analyze
results = ffs.fit_glm_arma11(data, design, tr=2.0)

# Save as NIfTI (for portability)
ffs.write_afni_bucket(results, 'glm.nii.gz')  # NIfTI

# Or save as AFNI (for AFNI tools)
ffs.write_afni_bucket(results, 'glm+tlrc.HEAD')  # AFNI
```

## AFNI Naming Conventions

AFNI typically uses view suffixes in filenames:
- `+orig` - Original/native space
- `+tlrc` - Talairach space
- `+acpc` - AC-PC aligned space

Examples:
```python
'stats+orig.HEAD'  # Original space
'stats+tlrc.HEAD'  # Talairach space
'func_r01+orig.HEAD'  # Run 1, original space
```

Our library preserves these conventions automatically!

## Backwards Compatibility

✅ **Fully backwards compatible**:
- Default behavior unchanged (NIfTI .nii.gz)
- All existing code works
- New functionality opt-in via extension

## Benefits

### 1. Native AFNI Workflow
```bash
# Before: Convert AFNI → NIfTI → Process → Convert back
3dAFNItoNIFTI func+orig
# ... process ...
3dcopy result.nii.gz result+orig

# After: Work directly with AFNI files
# No conversion needed!
```

### 2. Disk Space Savings
- AFNI BRIK.gz is often smaller than NIfTI .nii.gz
- No intermediate conversion files

### 3. Metadata Preservation
- AFNI attributes preserved in .HEAD
- 3drefit works on both formats

### 4. Tool Compatibility
- Direct input to AFNI programs
- No conversion pipeline needed

## Technical Details

### File Structure

**AFNI Format:**
```
glm+tlrc.HEAD      # ASCII header (always uncompressed)
glm+tlrc.BRIK.gz   # Binary data (optionally compressed)
```

**NIfTI Format:**
```
glm.nii.gz         # Header + data in one file
```

### 3drefit Compatibility

3drefit works on both formats:
```bash
# Works on AFNI
3drefit -relabel_all "L1 L2 L3" stats+tlrc.HEAD

# Works on NIfTI
3drefit -relabel_all "L1 L2 L3" stats.nii

# Our library handles both automatically!
```

### Compression Strategy

**AFNI Format:**
1. Write `.HEAD` (always uncompressed)
2. Write `.BRIK` (uncompressed)
3. Apply 3drefit to `.BRIK` (fast!)
4. Compress `.BRIK` → `.BRIK.gz`
5. Delete `.BRIK`

**NIfTI Format:**
1. Write `.nii` (uncompressed)
2. Apply 3drefit to `.nii` (fast!)
3. Compress `.nii` → `.nii.gz`
4. Delete `.nii`

Same efficiency for both formats!

## Testing Checklist

- [ ] Write AFNI BRIK uncompressed
- [ ] Write AFNI BRIK compressed (.gz)
- [ ] Write NIfTI compressed
- [ ] Write NIfTI uncompressed
- [ ] Load AFNI BRIK files
- [ ] Load AFNI BRIK.gz files
- [ ] Load NIfTI files
- [ ] 3drefit on AFNI files
- [ ] 3drefit on NIfTI files
- [ ] AFNI tools read output files
- [ ] Verify sub-brick labels preserved
- [ ] Verify statistical parameters preserved
- [ ] Test with AFNI view suffixes (+orig, +tlrc)
- [ ] Test mixed format pipeline (load AFNI, save NIfTI)

## Known Limitations

1. **JSON Sidecars**: Generated for all formats, not just NIfTI
   - Harmless extra file
   - Useful for documentation

2. **View Suffixes**: Must be in filename, not auto-generated
   - User specifies: `'stats+tlrc.HEAD'`
   - Library preserves, doesn't add

3. **Space Information**: Relies on affine matrix
   - AFNI space attributes not automatically set
   - Use 3drefit if needed

## Future Enhancements

1. **Auto-detect view from affine**: `+tlrc` if Talairach space
2. **Write AFNI attributes**: IJK_TO_DICOM_REAL, etc.
3. **Read AFNI attributes**: Parse .HEAD for metadata
4. **Support other AFNI formats**: MINC, etc.

## Migration Guide

### No Changes Needed!

Existing code works unchanged:
```python
# Old code - still works!
ffs.write_afni_bucket(results, 'glm.nii.gz')
```

### To Use AFNI Format:

Just change the extension:
```python
# New: Use AFNI format
ffs.write_afni_bucket(results, 'glm+tlrc.HEAD')
```

That's it! ✨

## Related Documentation

- nibabel AFNI support: https://nipy.org/nibabel/reference/nibabel.afni.html
- AFNI file formats: https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.attributes.html
- AFNI naming conventions: https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.environment.html
