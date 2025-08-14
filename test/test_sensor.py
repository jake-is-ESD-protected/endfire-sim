from endfiresim.wave import *
from endfiresim.sensor import *
from numpy import testing as npt

TEST_CARDIOID_POS0 = (0, 0, 0)
TEST_CARDIOID_AZIM0 = 0
TEST_CARDIOID_ELEV0 = 0
TEST_CARDIOID_AZIM_RANGE = np.linspace(0, np.pi, 3)
TEST_CARDIOID_ELEV1 = 0

TEST_FREQ = 1000
TEST_AMP = 0.5
TEST_C = 343
TEST_AZIM_PLANAR = 0
TEST_ELEV_PLANAR = 0

TEST_FS = 48000
TEST_DUR = 2 # s
TEST_TIME_STAMP = 0
TEST_TIME_FRAME = np.arange(0, TEST_DUR, 1/TEST_FS)


def test_sensor_init():
    try:
        s = CSensor()
    except TypeError:
        pass


def test_cardioid_ideal_init():
    ci = CCardioidIdeal(TEST_CARDIOID_POS0, TEST_CARDIOID_AZIM0, TEST_CARDIOID_ELEV0)
    assert ci.xyz == TEST_CARDIOID_POS0
    assert ci.azim == TEST_CARDIOID_AZIM0
    assert ci.elev == TEST_CARDIOID_ELEV0


def test_cardioid_ideal_vec():
    expected_vecs = ([1., 0., 0.], [0., 1., 0.], [-1., 0., 0.])

    for i, angle in enumerate(TEST_CARDIOID_AZIM_RANGE):
        ci = CCardioidIdeal(TEST_CARDIOID_POS0, angle, TEST_CARDIOID_ELEV0)
        vec = ci._CCardioidIdeal__vec()
        assert vec.shape == (3,)
        npt.assert_allclose(vec, np.asarray(expected_vecs[i]), atol=1e-7)


def test_cardioid_ideal_receive_back():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    ci = CCardioidIdeal(TEST_CARDIOID_POS0, TEST_AZIM_PLANAR, TEST_CARDIOID_ELEV0)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(p, np.zeros_like(p))
    npt.assert_allclose(gain, 0.)


def test_cardioid_ideal_receive_front():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    ci = CCardioidIdeal(TEST_CARDIOID_POS0, TEST_AZIM_PLANAR + np.pi, TEST_CARDIOID_ELEV0)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(np.max(np.real(p)), pw.amp)
    npt.assert_allclose(gain, 1.)


def test_cardioid_ideal_receive_side():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_AZIM_PLANAR, TEST_ELEV_PLANAR)
    ci = CCardioidIdeal(TEST_CARDIOID_POS0, TEST_AZIM_PLANAR + np.pi/2, TEST_CARDIOID_ELEV0)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(np.max(np.real(p)), pw.amp/2)
    npt.assert_allclose(gain, 0.5)