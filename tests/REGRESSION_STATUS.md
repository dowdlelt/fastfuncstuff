# AFNI Regression Test Status

## Test Data
- **Location**: `~/Dropbox/Data/small_validation_afni_data/`
- **Runs**: 2 runs, 10x10x10 voxels, 360 TRs each
- **Design**: 2 stimuli (movie, prompt) + 8 polynomials
- **GLT**: movieVprompt (movie - prompt)

## Results Summary

### ✅ OLS: **PERFECT MATCH**

All comparisons pass with <1% relative error:

| Metric | Max Abs Diff | Max Rel Diff | Correlation | Status |
|--------|--------------|--------------|-------------|--------|
| Full F-stat | 6.0e-05 | 0.58% | 1.000 | ✓ PASS |
| movie Coef | - | <1% | 1.000 | ✓ PASS |
| movie T-stat | - | <1% | 1.000 | ✓ PASS |
| prompt Coef | - | <1% | 1.000 | ✓ PASS |
| prompt T-stat | - | <1% | 1.000 | ✓ PASS |

**Conclusion**: OLS implementation is numerically identical to AFNI! 🎉

### ⚠️  REML: **SIGNIFICANT DIFFERENCES**

| Metric | Max Abs Diff | Max Rel Diff | Correlation | Status |
|--------|--------------|--------------|-------------|--------|
| Full F-stat | 31.1 | 49,101% | 0.757 | ✗ FAIL |

**Issue**: F-statistics differ substantially between our REML and AFNI's 3dREMLfit.

**Possible causes**:
1. Different ARMA parameters selected during grid search
2. Different F-stat calculation for REML
3. Bug in our F-stat computation with prewhitened data
4. Different degrees of freedom calculation

**Next steps**:
1. Compare ARMA parameters (a, b, lambda) between implementations
2. Check if same (a,b) values give same F-stats
3. Verify F-stat formula for REML case

### 📊 ARMA Parameters: **NOT YET TESTED**

Test infrastructure ready but needs investigation after F-stat issue resolved.

## Known Issues

### 1. GLTs Not Attached to OLS Results
- GLTs are computed (message says "Computed 1 GLT contrasts")
- But `results.glt_contrasts` attribute doesn't exist
- Not written to bucket file
- **TODO**: Fix GLT attachment to results object

### 2. REML F-stats Don't Match AFNI
- Large discrepancies (>30 in F-value)
- Correlation only 0.76 (should be >0.99)
- Primary blocking issue for REML validation

## Test Coverage

```bash
# Run all regression tests
pytest tests/test_afni_regression.py -v

# Run specific test
pytest tests/test_afni_regression.py::test_ols_matches_afni -v -s
```

### Current Status: 1/3 passing
- ✅ `test_ols_matches_afni` - PASS
- ❌ `test_reml_matches_afni` - FAIL (F-stat mismatch)
- ❌ `test_arma_params_match_afni` - FAIL (needs F-stat fix first)

## Grid Configuration

ARMA grid now matches AFNI exactly:
- **a_grid**: [0.0, 0.1, 0.2, ..., 0.9] (10 points)
- **b_grid**: [-0.9, -0.8, ..., 0.8, 0.9] (19 points)
- **Total**: 190 combinations (filtered by stability)

## Validation Metrics

Our comparison function reports:
- Mean difference (μ)
- Standard deviation (σ)
- Max absolute difference
- Max relative difference (%)
- Pearson correlation

**Acceptance criteria**:
- Correlation > 0.999
- Max rel diff < 1% for most values
- Mean diff ≈ 0

## Future Work

1. **Fix REML F-stat** - Primary blocker
2. **Validate ARMA parameters** - Compare (a,b,λ) selection
3. **Add GLT support** - Fix GLT attachment to results
4. **Add more datasets** - Test with different designs, more voxels
5. **CI Integration** - Add to GitHub Actions
