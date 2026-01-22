# Type Checking & Linting Workflow

This project uses **pyright** for type checking and **ruff** for linting to catch errors before runtime.

## Quick Start

```bash
# Run all checks before committing
make check

# Auto-fix linting issues
make fix

# Just type checking
make check-types

# Just linting
make check-lint
```

## What Gets Caught

### Pyright (Type Checker)
- ✅ **Import errors**: `from module import nonexistent_function`
- ✅ **Function signature errors**: Wrong parameters, missing arguments
- ✅ **Undefined variables**: Using variables before assignment
- ✅ **Type mismatches**: Passing wrong types to functions

**Example errors caught:**
```python
# ❌ reportCallIssue - missing required parameter
fit_glm(data=X, design=Y)  # Missing 'tr' parameter

# ❌ reportCallIssue - nonexistent parameter
fit_glm(data=X, design=Y, tr=2.0, mode='invalid')

# ❌ reportMissingImports
from fastfuncsim.glm_core import fit_glm_torch  # Doesn't exist
```

### Ruff (Linter)
- ✅ **F errors**: Undefined names, unused imports, syntax errors
- ✅ **E errors**: Code style (mostly auto-fixable)
- ✅ **B errors**: Likely bugs (mutable defaults, etc.)
- ✅ **I errors**: Import sorting

## Integration with Development

### Before Committing
```bash
# Run checks
./check_types.sh
# or
make check

# Fix auto-fixable issues
make fix
```

### In Your Editor

**VS Code** (recommended):
```json
// .vscode/settings.json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.linting.enabled": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true,
      "source.fixAll": true
    }
  }
}
```

**Vim/Neovim**: Use ALE or LSP with pyright + ruff

## Configuration Files

- `pyrightconfig.json` - Type checking rules
- `pyproject.toml` - Ruff linting rules + tool config
- `check_types.sh` - Manual check script
- `Makefile` - Convenient make targets

## Common Errors & Fixes

### Import Errors
```python
# ❌ Wrong
from fastfuncsim.glm_core import fit_glm_torch

# ✅ Correct - check the actual function name
from fastfuncsim.glm_core import fit_glm
```

### Function Signature Errors
```python
# ❌ Wrong - missing required parameters
hrf = get_canonical_hrf(mode='spmg1', tr=2.0)

# ✅ Correct - check function signature
hrf = get_canonical_hrf(stim_duration=0.0, tr=2.0, duration=32.0)
```

### Parameter Name Errors
```python
# ❌ Wrong - parameter doesn't exist
results = fit_glm(data=X, design=Y, nuisance=Z, return_residuals=True)

# ✅ Correct - check parameter names
results = fit_glm(data=X, design=Y, nuisance=Z)
```

## Pro Tips

1. **Run checks frequently** - Don't wait until you have 100 errors
2. **Focus on "error" level first** - Warnings can wait
3. **Key error types to fix**:
   - `reportCallIssue` - Wrong function calls
   - `reportArgumentType` - Wrong types passed
   - `reportMissingImports` - Import errors
   - `reportUndefinedVariable` - Typos/undefined names

4. **Auto-fix what you can**: `make fix` handles formatting and imports

## Example Workflow

```bash
# Starting new feature
git checkout -b my-feature

# Write code...
vim fastfuncsim/my_module.py

# Check types as you go
make check-types

# Fix issues
# ... edit code ...

# Auto-fix formatting/imports
make fix

# Final check before commit
make check

# Commit
git add -A
git commit -m "Add my feature"
```

## Continuous Integration

Add to your CI pipeline:
```bash
# In GitHub Actions / GitLab CI
pip install pyright ruff
make check
```

This catches errors before they hit production!
