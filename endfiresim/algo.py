import numpy as np


def sph_to_cart_2d(r: float | np.ndarray, azim: float | np.ndarray):
    x = r * np.cos(azim)
    y = r * np.sin(azim)
    return x, y


def sph_to_cart_3d(r: float | np.ndarray, elev: float | np.ndarray, azim: float | np.ndarray):
    x = np.array(r * np.sin(elev) * np.cos(azim))
    y = np.array(r * np.sin(elev) * np.sin(azim))
    z = np.array(-r * np.cos(elev))
    return x, y, z


def cart_to_sph_2d(x: float | np.ndarray, y: float | np.ndarray):
    r = np.sqrt(x**2 + y**2)
    azim = np.arctan2(y, x)
    azim = np.mod(azim, 2*np.pi)
    azim = np.where(r == 0, 0, azim)
    return r, azim


def cart_to_sph_3d(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray):
    r = np.sqrt(x**2 + y**2 + z**2)
    safe_r = np.where(r == 0, 1e-10, r)
    elev = np.arccos(-z / safe_r)
    azim = np.arctan2(y, x)
    azim = np.mod(azim, 2*np.pi)
    elev = np.where(r == 0, 0, elev)
    azim = np.where(r == 0, 0, azim)
    return r, elev, azim