from endfiresim.algo import *
from numpy import testing as npt
import numpy as np


POINTS_CART_2D = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1)
)

POINT_SPH_2D = (
    (1, 0),
    (1, np.pi),
    (1, np.pi/2),
    (1, 3*np.pi/2)
)


POINTS_CART_3D = (
    (0, 0, 1),
    (0, 0, -1),
    (1, 0, 0),
    (-1, 0, 0)
)

POINT_SPH_3D = (
    (1, 0, 0),
    (1, np.pi, 0),
    (1, np.pi/2, 0),
    (1, np.pi/2, np.pi)
)


def test_sph_to_cart_2d():
    for cart, sph in zip(POINTS_CART_2D, POINT_SPH_2D):
        x, y = sph_to_cart_2d(sph[0], sph[1])
        npt.assert_allclose(x, cart[0], atol=1e-7)
        npt.assert_allclose(y, cart[1], atol=1e-7)


def test_sph_to_cart_3d():
    for cart, sph in zip(POINTS_CART_3D, POINT_SPH_3D):
        x, y, z = sph_to_cart_3d(sph[0], sph[1], sph[2])
        npt.assert_allclose(x, cart[0], atol=1e-7)
        npt.assert_allclose(y, cart[1], atol=1e-7)
        npt.assert_allclose(z, cart[2], atol=1e-7)


def test_sph_to_cart_2d():
    for cart, sph in zip(POINTS_CART_2D, POINT_SPH_2D):
        r, azim = cart_to_sph_2d(cart[0], cart[1])
        npt.assert_allclose(r, sph[0], atol=1e-7)
        npt.assert_allclose(azim, sph[1], atol=1e-7)


def test_sph_to_cart_3d():
    for cart, sph in zip(POINTS_CART_3D, POINT_SPH_3D):
        r, elev, azim = cart_to_sph_3d(cart[0], cart[1], cart[2])
        npt.assert_allclose(r, sph[0], atol=1e-7)
        npt.assert_allclose(elev, sph[1], atol=1e-7)
        npt.assert_allclose(azim, sph[2], atol=1e-7)