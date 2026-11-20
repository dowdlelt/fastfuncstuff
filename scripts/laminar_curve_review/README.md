# Laminar timing and model audits

These CPU audits reproduce the synthetic experiments in the
[laminar validation report](../../reports/Laminar%20timing%20and%20model%20validation.md).
Run from the repository root with fastfuncstuff installed.

```sh
python scripts/laminar_curve_review/tr_knots_audit.py --output tr_knots_audit.json
python scripts/laminar_curve_review/observation_operator.py
python scripts/laminar_curve_review/physiology_audit.py
python scripts/laminar_curve_review/integration_audit.py
```

The timing audit also runs an independent AFNI comparison if `3dDeconvolve`
is available. All four audits generate their synthetic inputs in code.
