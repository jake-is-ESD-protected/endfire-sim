from endfiresim.wave import *
from endfiresim.sensor import *
from numpy import testing as npt

TEST_SENSOR_POS = (0, 0, 0)
TEST_CARDIOID_AZIM0 = 0
TEST_CARDIOID_ELEV0 = np.pi/2
TEST_CARDIOID_AZIM_RANGE = np.linspace(0, np.pi, 3)
TEST_CARDIOID_ELEV1 = 0

TEST_FREQ = 1000
TEST_AMP = 0.5
TEST_C = 343
TEST_AZIM_PLANAR = 0
TEST_ELEV_PLANAR = np.pi/2

TEST_FS = 48000
TEST_DUR = 2 # s
TEST_TIME_STAMP = 0
TEST_TIME_FRAME = np.arange(0, TEST_DUR, 1/TEST_FS)

TEST_MONOPOLE_POS1 = (0, 0, 0)
TEST_MONOPOLE_POS2 = (0.02, 0.02, 0)
TEST_MONOPOLE_DIST = 0.02


def test_sensor_init():
    s = CSensor(TEST_SENSOR_POS)
    assert s.xyz == TEST_SENSOR_POS


def test_sensor_receive():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    s = CSensor(TEST_SENSOR_POS)
    p, gain = s.receive(pw, TEST_TIME_FRAME)
    assert gain == 1.
    npt.assert_allclose(p, pw.p(TEST_TIME_FRAME, s.xyz))


def test_cardioid_ideal_init():
    ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0, TEST_CARDIOID_AZIM0)
    assert ci.xyz == TEST_SENSOR_POS
    assert ci.elev == TEST_CARDIOID_ELEV0
    assert ci.azim == TEST_CARDIOID_AZIM0


def test_cardioid_ideal_vec():
    expected_vecs = ([1., 0., 0.], [0., 1., 0.], [-1., 0., 0.])

    for i, angle in enumerate(TEST_CARDIOID_AZIM_RANGE):
        ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0, angle)
        vec = ci.direction_vec()
        assert np.shape(vec) == (3,)
        npt.assert_allclose(np.array(vec), np.asarray(expected_vecs[i]), atol=1e-7)


def test_cardioid_ideal_receive_back():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0, TEST_AZIM_PLANAR)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(p, np.zeros_like(p))
    npt.assert_allclose(gain, 0.)


def test_cardioid_ideal_receive_front():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0, TEST_AZIM_PLANAR + np.pi)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)
    npt.assert_allclose(gain, 2.)


def test_cardioid_ideal_receive_side():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0, TEST_AZIM_PLANAR + np.pi/2)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)
    npt.assert_allclose(gain, np.sqrt(2.))


def test_cardioid_ideal_receive_top():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C,  TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    ci = CCardioidIdeal(TEST_SENSOR_POS, TEST_CARDIOID_ELEV0 + np.pi/2, TEST_AZIM_PLANAR)
    p, gain = ci.receive(pw, TEST_TIME_FRAME)
    assert p.shape == TEST_TIME_FRAME.shape
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)
    npt.assert_allclose(gain, np.sqrt(2.))


def test_cardioid_endfire_init_single_pos_dist():
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   distance=TEST_MONOPOLE_DIST,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_CARDIOID_AZIM0)
    npt.assert_allclose(cef.xyz, np.asarray(TEST_SENSOR_POS))
    pos2 = (TEST_SENSOR_POS[0] + TEST_MONOPOLE_DIST, TEST_SENSOR_POS[1], TEST_SENSOR_POS[2])
    npt.assert_allclose(cef.poss[1], np.asarray(pos2), atol=1e-7)
    assert cef.elev == TEST_CARDIOID_ELEV0
    assert cef.azim == TEST_CARDIOID_AZIM0
    assert cef.distance == TEST_MONOPOLE_DIST
    assert cef.freq == TEST_C / (4* cef.distance)
    assert cef.path_delay == cef.distance / TEST_C


def test_cardioid_endfire_init_single_pos_freq():
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_CARDIOID_AZIM0)
    npt.assert_allclose(cef.xyz, np.asarray(TEST_SENSOR_POS))
    assert cef.distance == TEST_C / (4 * TEST_FREQ)
    pos2 = (TEST_SENSOR_POS[0] + cef.distance, TEST_SENSOR_POS[1], TEST_SENSOR_POS[2])
    npt.assert_allclose(cef.poss[1], np.asarray(pos2), atol=1e-7)
    assert cef.elev == TEST_CARDIOID_ELEV0
    assert cef.azim == TEST_CARDIOID_AZIM0
    assert cef.freq == TEST_FREQ
    assert cef.path_delay == cef.distance / TEST_C


def test_cardioid_endfire_receive_back():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 0.0, atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain, atol=1e-7)


def test_cardioid_endfire_receive_front():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR + np.pi)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 2., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_cardioid_endfire_receive_side():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR + np.pi/2)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, np.sqrt(2.), atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_cardioid_endfire_receive_top():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0 + np.pi/2,
                   azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, np.sqrt(2.), atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_n3_endfire_receive_back():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=3,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 1., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain, atol=1e-7)


def test_n3_endfire_receive_front():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=3,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR + np.pi)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 3., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_n3_endfire_receive_side():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=3,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_AZIM_PLANAR + np.pi/2)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 1., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_n3_endfire_receive_top():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CEndfire(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=3,
                   elev=TEST_CARDIOID_ELEV0 + np.pi/2,
                   azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 1., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_cardioid_synthetic_init_single_pos_dist():
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                             distance=TEST_MONOPOLE_DIST,
                             n_sensors=2,
                             elev=TEST_CARDIOID_ELEV0,
                             azim=TEST_CARDIOID_AZIM0)
    npt.assert_allclose(cef.xyz, np.asarray(TEST_SENSOR_POS))
    pos2 = (TEST_SENSOR_POS[0] + TEST_MONOPOLE_DIST, TEST_SENSOR_POS[1], TEST_SENSOR_POS[2])
    npt.assert_allclose(cef.poss[1], np.asarray(pos2), atol=1e-7)
    assert cef.elev == TEST_CARDIOID_ELEV0
    assert cef.azim == TEST_CARDIOID_AZIM0
    assert cef.distance == TEST_MONOPOLE_DIST
    assert cef.freq == TEST_C / (4* cef.distance)
    assert cef.path_delay == cef.distance / TEST_C


def test_cardioid_synthetic_init_single_pos_freq():
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                   target_freq=TEST_FREQ,
                   n_sensors=2,
                   elev=TEST_CARDIOID_ELEV0,
                   azim=TEST_CARDIOID_AZIM0)
    npt.assert_allclose(cef.xyz, np.asarray(TEST_SENSOR_POS))
    assert cef.distance == TEST_C / (4 * TEST_FREQ)
    pos2 = (TEST_SENSOR_POS[0] + cef.distance, TEST_SENSOR_POS[1], TEST_SENSOR_POS[2])
    npt.assert_allclose(cef.poss[1], np.asarray(pos2), atol=1e-7)
    assert cef.elev == TEST_CARDIOID_ELEV0
    assert cef.azim == TEST_CARDIOID_AZIM0
    assert cef.freq == TEST_FREQ
    assert cef.path_delay == cef.distance / TEST_C


def test_cardioid_synthetic_receive_back():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                             target_freq=TEST_FREQ,
                             n_sensors=2,
                             elev=TEST_CARDIOID_ELEV0,
                             azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 0.0, atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain, atol=1e-7)


def test_cardioid_synthetic_receive_front():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                             target_freq=TEST_FREQ,
                             n_sensors=2,
                             elev=TEST_CARDIOID_ELEV0,
                             azim=TEST_AZIM_PLANAR + np.pi)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, 2., atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_cardioid_synthetic_receive_side():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                             target_freq=TEST_FREQ,
                             n_sensors=2,
                             elev=TEST_CARDIOID_ELEV0,
                             azim=TEST_AZIM_PLANAR + np.pi/2)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, np.sqrt(2.), atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)


def test_cardioid_synthetic_receive_top():
    pw = CWaveModelPlanar(TEST_FREQ, TEST_AMP, TEST_C, TEST_ELEV_PLANAR, TEST_AZIM_PLANAR)
    cef = CCardioidSynthetic(xyz=TEST_SENSOR_POS,
                             target_freq=TEST_FREQ,
                             n_sensors=2,
                             elev=TEST_CARDIOID_ELEV0 + np.pi/2,
                             azim=TEST_AZIM_PLANAR)
    p, gain = cef.receive(pw, TEST_TIME_FRAME)
    npt.assert_allclose(gain, np.sqrt(2.), atol=1e-7)
    npt.assert_allclose(np.max(np.real(p)), pw.amp * gain)