Scripts behind `../fmri_wiki/concepts/Response curves for the laminar model.md` (2026-11-10).
One-off experiment code, kept as the record of what was run; paths are the session's.

- `cmp.py`, `run_orig.m`, `make_cbf.py`, `const.py` — our vascular model against Havlicek's 2020
  `LBR_model.m` and analytic steady state; the `eps_v` typo and TE sensitivity.
- `gen.py` — ground-truth laminar time series on the real sub-3010 ses-fine event timing.
- `ev2.py`, `proj2.py` — estimator bias/noise against truth; projection of the truth on the tents.
- `inv.py` — invert an estimator's curves through the model space.
- `nonlin.py`, `nonlin2.py` — the model's response to the real event train against a sum of isolated ones.
- `run.sh`, `show.py`, `scan.py` — real ROI x depth refits, and fits across estimator / latency / free parameters.
