# Outstanding Issues

---

## 1. MOCO motion parameter sign convention (pitch, yaw, dL, dP)

### Symptom
Benchmark validation shows four of the six AFNI-format `.1D` motion parameters from
`ffs_moco` are **anti-correlated** (~−0.97) with the corresponding AFNI `3dvolreg` output,
while the other two are correctly correlated:

| Column | AFNI name | ffs expression | Benchmark r |
|--------|-----------|----------------|-------------|
| 1 | roll  | `−rz_DICOM` | +0.94  ✓ |
| 2 | pitch | `+rx_DICOM` | −0.97  ✗ |
| 3 | yaw   | `+ry_DICOM` | −0.97  ✗ |
| 4 | dS    | `−dz_DICOM` | +0.998 ✓ |
| 5 | dL    | `+dx_DICOM` | −0.97  ✗ |
| 6 | dP    | `+dy_DICOM` | −0.92  ✗ |

The aligned volumes themselves agree excellently (mean-image r ≥ 0.9998), so this is a
parameter-output bug, not an alignment bug.

### Root cause analysis

The pipeline that produces the `.1D` values is:

```
voxel-space matrix (M_ijk, base→source)
  → voxel_matrix_to_dicom()   [affine.py, function voxel_matrix_to_dicom]
  → matrix_to_params()        [affine.py, function matrix_to_params]
  → save_moco_1D()            [ffs_moco.py, function save_moco_1D]
```

**Key asymmetry introduced by the RAS→DICOM conversion.**
`voxel_matrix_to_dicom` applies `D = diag(−1, −1, 1, 1)` on both sides
(`D @ M_ras @ D`). This negates x and y — but NOT z — in both the rotation and
translation parts of the matrix:

- `dx_DICOM = −tx_RAS`
- `dy_DICOM = −ty_RAS`
- `dz_DICOM = +tz_RAS`  ← no sign flip

Because the transform matrix corrects motion (image-warp direction), and the AFNI `.1D`
convention reports the motion itself (subject-displacement direction), every component
needs one negation. The z-axis gets this negation only from `save_moco_1D` (which
writes `−dz`, `−rz`). The x,y axes get the D-matrix negation during conversion but
*don't* get a second negation in `save_moco_1D`, so they end up with the wrong sign.

**Hypothesis for fix** (in `save_moco_1D` and `save_moco_dfile`):

```python
# Likely correct mapping:
roll  = −rz_DICOM   # unchanged (already correct)
pitch = −rx_DICOM   # was +rx_DICOM
yaw   = −ry_DICOM   # was +ry_DICOM
dS    = −dz_DICOM   # unchanged (already correct)
dL    = −dx_DICOM   # was +dx_DICOM
dP    = −dy_DICOM   # was +dy_DICOM
```

### What needs verification before fixing

The sign convention for AFNI's `.1D` pitch, yaw, dL, dP axes is not self-evident from
the AFNI docs alone. The cleanest verification is to:

1. Create a synthetic NIfTI volume shifted exactly +2 mm in the x voxel direction.
2. Run `3dvolreg -base 0 -1Dfile out.1D shifted.nii.gz`.
3. Check the sign of the dL column in `out.1D`.
4. Run the same through `ffs_moco` and compare.

If dL is positive in both, the fix above is correct.  
If AFNI writes negative dL for a leftward shift, our current code is actually correct and
the issue is elsewhere (e.g. rotation angle extraction convention in `matrix_to_params`).

### Code locations with TODOs

TODOs have been added at the three most likely fault points (search for
`TODO(sign-convention)` across the codebase):

- **`ffs_moco.py`, function `save_moco_1D`** — the column mapping from DICOM params to
  AFNI `.1D` format. Primary suspect. Contains the full hypothesis comment.
- **`ffs_moco.py`, function `save_moco_dfile`** — identical mapping, same fix needed
  once `save_moco_1D` is resolved.
- **`affine.py`, function `voxel_matrix_to_dicom`** — where `D @ M_ras @ D` introduces
  the asymmetric x,y sign flip. Not necessarily wrong itself, but understanding this is
  required to verify the fix.

The rotation angle extraction in `affine.py` `matrix_to_params` is a secondary suspect
(the Rz@Rx@Ry decomposition convention may differ from AFNI's), but the translational
anti-correlations (dL, dP) point more directly at the `save_moco_1D` mapping.

## ICA Temporal Concat differences
Noting here that the the number of components is very different when using temp concat. 
This could be due to different masking, or the data reduction that MELODIC performs, or both. 
Further investigation is required. Components themselves look similar, across 50 to 60%, so its not terrible at this momment. 

## ICA No Tensorial Approach currently available. 
For datasets with the same design (or multiecho dataset) the tensor approach is valid. 
This is currently not implemente. 

## Parametric Duration Modulation (future feature)

Per-event HRF amplitude scaling by event duration — i.e., each event's HRF is scaled
by that event's individual duration rather than treating all events of a condition as
identical.

### What this would look like
BIDS events TSV already provides per-event `duration` values. The feature would use
these to scale the HRF amplitude (or boxcar height) for each individual event before
convolution with the canonical HRF. This is distinct from the current approach where
all events of a condition share one duration (derived from the unique values in the TSV).

### Why it is not implemented yet
The current pipeline works at the condition level: `all_onsets[cond][run]` is an array
of onset times, and `durations[cond]` is a single scalar. Supporting per-event
modulation requires threading per-event duration values through:
- `create_onset_matrix_microtime` (currently takes a scalar duration per condition)
- `build_glm_design` / `build_single_trial_design` (same assumption)
- The HRF convolution step

This is a moderate refactor — the data structures need to change from scalar durations
to arrays-of-durations — and should be done carefully to avoid breaking the existing
API surface.

### Code locations to modify
- `fastfuncstuff/design/builder.py` — `create_onset_matrix_microtime`: accept per-event
  duration arrays in addition to scalar duration
- `fastfuncstuff/design/bids_events.py` — `parse_bids_events`: return per-event durations
  alongside the per-condition median (already stores them internally in `cond_dur_sets`)
- All CLIs that accept `-events` — thread per-event durations through to the design builder