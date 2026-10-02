"""InstaPCA: the engine's arithmetic, and the review loop that ends in an ortvec."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import instapca as engine
from fastfuncstuff.viewer.ortvec import read_columns
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import SetMode, SetOverlay, SetUnderlay

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
TR = 2.0
N_TIME = 80
SHAPE = (10, 10, 8)
AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])


def _planted(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A bright box carrying two components, each confined to its own corner,
    on a drift every voxel shares. Returns the data, the box, and both sources."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIME)
    slow = np.sin(2 * np.pi * t / 23.0)
    fast = np.sign(np.sin(2 * np.pi * t / 7.0))
    data = rng.normal(5.0, 0.5, (*SHAPE, N_TIME)).astype(np.float32)
    box = np.zeros(SHAPE, bool)
    box[1:9, 1:9, 1:7] = True
    data[box] += 1000.0 + np.linspace(0, 30, N_TIME) + rng.normal(0, 1.0, (box.sum(), N_TIME))
    # Different sizes, so the two eigenvalues are well apart: equal ones would
    # leave PCA free to hand back any rotation of the pair.
    data[1:5, 1:5, 1:5] += 8.0 * slow
    data[6:8, 6:8, 4:6] += 6.0 * fast
    return data, box, slow, fast


def _r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# -- engine -----------------------------------------------------------------


def test_the_planted_components_lead_with_positive_maps_where_they_were_put():
    data, box, slow, fast = _planted()
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, n_components=5, device=CPU)
    assert d.n_components == 5
    picked = []
    for source, where in ((slow, (3, 3, 3)), (fast, (6, 6, 4))):
        fits = [abs(_r(d.timecourses[:, j], source)) for j in range(2)]
        k = int(np.argmax(fits))
        assert fits[k] > 0.9
        picked.append(k)
        # Sign anchored on the map's heavy tail: the corner the component lives
        # in is positive, whatever sign eigh handed back.
        assert d.volume(k)[where] > 0.8
        assert np.sum(d.correlation[:, k].astype(float) ** 3) > 0
    assert sorted(picked) == [0, 1]


def test_the_map_is_the_pearson_correlation_with_the_detrended_voxel():
    """The whole interpretability claim: a number in [-1, 1] that means the
    same thing in every voxel. Checked against a plain numpy correlation."""
    from fastfuncstuff.viewer.derive import orthonormal_basis, read_nuisance

    data, box, _slow, _fast = _planted(1)
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, polort=2, device=CPU)
    q = orthonormal_basis(read_nuisance(None, n_time=N_TIME, polort=2).columns)
    voxel = data[3, 3, 3].astype(np.float64)
    detrended = voxel - q @ (q.T @ voxel)
    for k in range(3):
        assert d.volume(k)[3, 3, 3] == pytest.approx(_r(detrended, d.timecourses[:, k]), abs=1e-4)
    assert np.nanmax(np.abs(d.correlation)) <= 1.0 + 1e-5
    assert np.isnan(d.volume(0)[0, 0, 0])


def test_the_gram_spectrum_matches_an_svd_of_the_prepared_matrix():
    from fastfuncstuff.viewer.derive import orthonormal_basis, read_nuisance

    data, box, _slow, _fast = _planted(2)
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, polort=1, n_components=6, device=CPU)
    q = orthonormal_basis(read_nuisance(None, n_time=N_TIME, polort=1).columns)
    y = data[box].astype(np.float64)
    y = y - (y @ q) @ q.T
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    s = np.linalg.svd(y, compute_uv=False)
    np.testing.assert_allclose(d.explained, (s**2 / np.sum(s**2))[:6], rtol=1e-4)


def test_timecourses_are_unit_variance_and_orthogonal_to_the_drift():
    data, box, _slow, _fast = _planted(3)
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, polort=2, device=CPU)
    np.testing.assert_allclose(d.timecourses.std(axis=0), 1.0, rtol=1e-6)
    linear = np.linspace(-1, 1, N_TIME)
    cosine = (linear @ d.timecourses) / (np.linalg.norm(linear) * np.sqrt(N_TIME))
    assert np.max(np.abs(cosine)) < 1e-4  # float32 maps, float64 eigh
    np.testing.assert_allclose(d.timecourses.mean(axis=0), 0.0, atol=1e-3)


def test_amplitude_is_the_components_share_in_percent_of_the_mean():
    """A voxel that is a mean plus exactly the component: amplitude is the
    component's standard deviation over that mean, and correlation is 1."""
    data, box, slow, _fast = _planted(4)
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, n_components=3, device=CPU)
    k = int(np.argmax([abs(_r(d.timecourses[:, j], slow)) for j in range(3)]))
    tc = d.timecourses[:, k]
    data[0, 0, 0] = 200.0 + 5.0 * tc
    mask = box.copy()
    mask[0, 0, 0] = True
    d2 = engine.decompose(data, affine=AFFINE, tr=TR, mask=mask, n_components=3, device=CPU)
    j = int(np.argmax([abs(_r(d2.timecourses[:, i], tc)) for i in range(3)]))
    assert abs(d2.volume(j)[0, 0, 0]) == pytest.approx(1.0, abs=0.02)
    assert abs(d2.volume(j, "amplitude")[0, 0, 0]) == pytest.approx(100 * 5.0 / 200.0, rel=0.03)


def test_the_ortvec_reads_back_through_the_ortvec_reader(tmp_path):
    data, box, _slow, _fast = _planted(5)
    d = engine.decompose(data, affine=AFFINE, tr=TR, mask=box, n_components=6, device=CPU)
    path = engine.write_ortvec(tmp_path / "noise.1D", d, [4, 1], source="bold")
    columns, _names = read_columns(str(path))
    assert columns.shape == (N_TIME, 2)
    np.testing.assert_allclose(columns, d.timecourses[:, [1, 4]], atol=1e-5)
    assert path.read_text().startswith("# InstaPCA noise components of bold: 1 4")


def test_a_mask_on_another_grid_is_refused(tmp_path):
    p = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), AFFINE), str(p))
    with pytest.raises(ValueError, match="resample"):
        engine.read_mask(p, SHAPE)


# -- mode -------------------------------------------------------------------


def _write(path, data, tr=0.0):
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), AFFINE)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


@pytest.fixture
def pca_session(tmp_path):
    data, box, _slow, _fast = _planted(6)
    _write(tmp_path / "anat.nii.gz", data.mean(axis=3))
    _write(tmp_path / "bold.nii.gz", data, tr=TR)
    _write(tmp_path / "mask.nii.gz", box.astype(np.float32))
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    s.tmp = tmp_path
    yield s
    s.close()


def test_label_noise_and_save_writes_the_noise_columns_beside_the_run(pca_session):
    s = pca_session
    s.do(SetMode("instapca"))
    mode = s.mode
    assert mode._result is not None
    assert s.state.layers.find_by_source("mode:instapca") is not None
    mode.action("noise")  # PC 0, steps to 1
    mode.action("signal")  # PC 1, steps to 2
    mode.action("noise")  # PC 2
    assert mode.noise_components() == [0, 2]
    mode.action("save")
    out = s.tmp / "bold_pca_noise.1D"
    columns, _ = read_columns(str(out))
    np.testing.assert_allclose(columns, mode._result.timecourses[:, [0, 2]], atol=1e-5)


def test_saving_with_nothing_labelled_noise_says_so(pca_session):
    pca_session.do(SetMode("instapca"))
    with pytest.raises(ValueError, match="noise"):
        pca_session.mode.action("save")


def test_more_components_keep_labels_and_a_new_polort_drops_them(pca_session):
    s = pca_session
    s.do(SetMode("instapca"))
    s.mode.labels = {0: "noise", 9: "noise"}
    s.set_mode_param("n_components", "5")
    assert s.mode.labels == {0: "noise"}
    s.set_mode_param("polort", "3")
    assert s.mode.labels == {}


def test_coming_back_to_the_mode_does_not_redecompose(pca_session):
    s = pca_session
    s.do(SetMode("instapca"))
    first = s.mode._result
    s.do(SetMode("plain"))
    s.do(SetMode("instapca"))
    assert s.mode._result is first


def test_a_mask_file_restricts_the_decomposition(pca_session):
    s = pca_session
    s.do(SetMode("instapca"))
    auto = int(s.mode._result.mask.sum())
    s.set_mode_param("mask", str(s.tmp / "mask.nii.gz"))
    assert int(s.mode._result.mask.sum()) == 8 * 8 * 6
    assert s.mode._result.mask_source == "mask.nii.gz"
    assert auto > 0
