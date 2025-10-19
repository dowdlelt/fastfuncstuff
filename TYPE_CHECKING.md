# Type Checking Setup for fastfuncsim

## Quick Check Before Running
To avoid waiting 10 minutes for data to load only to hit a type error:

```bash
conda activate py312_movie_tasks
python check_types.py
```

This catches **critical** errors like:
- ✅ Import errors (`ModuleNotFoundError`, wrong module names)
- ✅ Missing function parameters
- ✅ Typos in function names
- ❌ Ignores noise (nibabel private imports, minor type mismatches)

## Full Type Check
For comprehensive checking (includes warnings):

```bash
conda activate py312_movie_tasks
pyright fastfuncsim/
```

## Configuration
- `pyrightconfig.json`: Configured to focus on **argument type errors** and **call issues**
- Suppresses noise from:
  - Private imports (nibabel internals)
  - Unknown types from external libraries
  - Attribute access on dynamically typed objects

## What We Fixed Today
1. **Import error**: `from .glm import fit_glm` → `from .glm_core import fit_glm`
   - **Would have been caught by**: `python check_types.py` ✅
   - **Cost**: 10+ minutes of data loading wasted ❌

2. **Missing parameter**: `want_ols` not in `fit_glm_arma11()` signature
   - **Would have been caught by**: `pyright` during development ✅
   - **Cost**: ~3 minutes to debug and fix ❌

3. **Batch size calculation**: Wrong formula caused GPU OOM
   - **Would NOT be caught by**: Type checking (logic error) ❌
   - **Requires**: Testing with real data + monitoring

## Best Practice Workflow
```bash
# 1. Quick check before running experiments
python check_types.py

# 2. If clear, run your script
python examples/analyze_taskforce_ses02_clean.py

# 3. For development: full check periodically
pyright fastfuncsim/ | grep "error:" | wc -l
```

## Error Count History
- Before fixes: **255 errors**
- After type annotations + config: **47 errors** in library
- Critical errors (would break at runtime): **0** ✅
