# Validation of sandbox-only test failures

The complete fast tier finished with **8617 passed, 4 failed, 111 skipped, 4 xfailed in 581.16 s**. Every failure was a `PermissionError` opening a local multiprocessing socket (`bind` or `setsockopt`), rather than a numerical assertion.

To distinguish a regression from environment restrictions, I archived the initial commit `39e17f15` into `/tmp/ffs-laminar-baseline-39e17f15` using `git archive`, without changing the repository or its environment. Both targeted commands used the same Python 3.14.6 runtime `/home/logan/miniconda3/envs/py314_ffs/bin/python` and torch 2.14.1+cu130.

Baseline, under the same sandbox:

```bash
cd /tmp/ffs-laminar-baseline-39e17f15
PYTHONPATH=/tmp/ffs-laminar-baseline-39e17f15 UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python -m pytest tests/test_laminar_cli.py tests/test_simulation_explore.py -m 'not slow and not gpu' -n 0
```

Result: **4 failed, 17 passed, 6 deselected in 90.84 s**. The same four failures reproduce on the initial commit:

- `tests/test_laminar_cli.py::test_run_jobs_returns_each_roi_its_own_inversions_in_order[2]`
- `tests/test_laminar_cli.py::test_run_jobs_raises_a_worker_failure`
- `tests/test_laminar_cli.py::test_run_jobs_survives_the_default_file_limit`
- `tests/test_simulation_explore.py::test_worker_pool_scores_exactly_like_the_serial_loop`

Current checkout, rerun outside the sandbox after automatic approval review permitted local multiprocessing socket use:

```bash
cd /home/logan/Dropbox/Resources/code/fastfuncstuff
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python -m pytest tests/test_laminar_cli.py tests/test_simulation_explore.py -m 'not slow and not gpu' -n 0
```

Current unrestricted result: **21 passed, 6 deselected in 96.73 s**. All four formerly failing worker-pool cases pass when their required local sockets are permitted. Combined with identical baseline failures under restriction, this isolates the four fast-tier failures to the sandbox environment.

No GPU tests or second full suite were run; no code changes were needed for the socket restrictions.
