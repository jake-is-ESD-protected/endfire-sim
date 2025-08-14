from endfiresim.wave import *
from numpy import testing as npt

TEST_FREQ = 1000
TEST_AMP = 0.5
TEST_C = 343

TEST_AZIM_PLANAR = 0
TEST_ELEV_PLANAR = 0

TEST_SRC_POS_SPHERIC = (0, 0, 0)

TEST_OBSERVER_POS = (3, 3, 3)
x = y = z = np.linspace(-5, 5, 50)
TEST_FIELD_POS = np.meshgrid(x, y, z)

TEST_FS = 48000
TEST_DUR = 2 # s
TEST_TIME_STAMP = 0
TEST_TIME_FRAME = np.arange(0, TEST_DUR, 1/TEST_FS)

def test_wave_init():
    try:
        w = CWaveModel()
    except TypeError:
        pass


def test_wave_planar_init():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    npt.assert_allclose(pw.omega, TEST_FREQ*np.pi*2)
    npt.assert_allclose(pw.k, TEST_FREQ*np.pi*2/TEST_C)
    assert pw.type == "planar"


def test_wave_spheric_init():
    sw = CWaveModelSpheric(TEST_FREQ, TEST_AMP, TEST_C, TEST_SRC_POS_SPHERIC)
    npt.assert_allclose(sw.omega, TEST_FREQ*np.pi*2)
    npt.assert_allclose(sw.k, TEST_FREQ*np.pi*2/TEST_C)
    assert sw.type == "spheric"


def test_wave_planar_p_point_step():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    p = pw.p(TEST_TIME_STAMP, TEST_OBSERVER_POS)
    assert p.dtype == np.complex128
    assert p.shape == ()


def test_wave_planar_p_point_frame():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    p = pw.p(TEST_TIME_FRAME, TEST_OBSERVER_POS)
    assert p.dtype == np.complex128
    assert p.shape == np.shape(TEST_TIME_FRAME)
    npt.assert_allclose(np.max(np.real(p)), pw.amp, atol=1e-3)
    npt.assert_allclose(np.mean(np.real(p)), 0, atol=1e-3)


def test_wave_planar_p_field_step():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    p = pw.p(TEST_TIME_STAMP, TEST_FIELD_POS)
    assert p.dtype == np.complex128
    assert p.shape == np.shape(TEST_FIELD_POS)[1:]
    npt.assert_allclose(np.max(np.real(p)), pw.amp, atol=1e-3)
    npt.assert_allclose(np.mean(np.real(p)), 0, atol=1e-2)


def test_wave_planar_p_field_frame():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    try:
        p = pw.p(TEST_TIME_FRAME, TEST_FIELD_POS)
    except ValueError:
        pass


def test_wave_spheric_p_point_frame():
    sw = CWaveModelSpheric(TEST_FREQ, TEST_AMP, TEST_C, TEST_SRC_POS_SPHERIC)
    p = sw.p(TEST_TIME_STAMP, TEST_OBSERVER_POS)
    assert p.dtype == np.complex128
    assert p.shape == ()


def test_wave_spheric_p_point_frame():
    sw = CWaveModelSpheric(TEST_FREQ, TEST_AMP, TEST_C, TEST_SRC_POS_SPHERIC)
    p = sw.p(TEST_TIME_FRAME, TEST_OBSERVER_POS)
    assert p.dtype == np.complex128
    assert p.shape == np.shape(TEST_TIME_FRAME)
    delta = np.asarray(TEST_OBSERVER_POS) - sw.source_xyz
    R = np.linalg.norm(delta, axis=0)
    npt.assert_allclose(np.max(np.real(p)), sw.amp/R, atol=1e-3)
    npt.assert_allclose(np.mean(np.real(p)), 0, atol=1e7)


def test_wave_spheric_p_field_step():
    sw = CWaveModelSpheric(TEST_FREQ, TEST_AMP, TEST_C, TEST_SRC_POS_SPHERIC)
    p = sw.p(TEST_TIME_STAMP, TEST_FIELD_POS)
    assert p.dtype == np.complex128
    assert p.shape == np.shape(TEST_FIELD_POS)[1:]


def test_wave_spheric_p_field_frame():
    sw = CWaveModelSpheric(TEST_FREQ, TEST_AMP, TEST_C, TEST_SRC_POS_SPHERIC)
    try:
        p = sw.p(TEST_TIME_FRAME, TEST_FIELD_POS)
    except ValueError:
        pass