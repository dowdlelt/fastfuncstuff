"""Pool blur for the model-order count must never draw on non-pool (criteria) voxels."""

import numpy as np
import torch

from fastfuncstuff.denoise.sequential import prepare_pool_for_model_order

CPU = torch.device("cpu")


def test_blur_stays_inside_the_pool():
    # The strongest form of "no criteria voxel contributes": the blurred pool is
    # IDENTICAL whether or not the criteria voxels carry a huge signal.
    shape = (28, 28, 28)
    n_vol = int(np.prod(shape))
    T = 40
    g = torch.Generator().manual_seed(0)
    data = torch.randn(n_vol, T, generator=g)
    crit = np.zeros(shape, bool)
    crit[13:15, :, :] = True
    crit_flat = torch.as_tensor(crit.reshape(-1))
    loud = data.clone()
    loud[crit_flat] += torch.sin(torch.linspace(0, 20, T)) * 1000.0
    pool = ~crit_flat
    geom = dict(volume_shape=shape, voxel_sizes=(2.0, 2.0, 2.0), mask_flat=None)
    kw = dict(nuisance_per_run=None, smooth_fwhm=8.0, device=CPU)
    quiet_out, r_q = prepare_pool_for_model_order(data, [0], pool, geom, **kw)
    loud_out, r_l = prepare_pool_for_model_order(loud, [0], pool, geom, **kw)
    assert quiet_out.shape == (int(pool.sum()), T)
    torch.testing.assert_close(loud_out, quiet_out)
    assert r_q == r_l and r_q[0] > 1.0
