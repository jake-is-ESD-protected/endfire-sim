import numpy as np


def sph_to_cart_2d(r: float | np.ndarray, azim: float | np.ndarray):
    x = r * np.cos(azim)
    y = r * np.sin(azim)
    return x, y


def sph_to_cart_3d(r: float |np.ndarray, elev:float |np.ndarray, azim: float |np.ndarray):
    x = np.array(np.sin(elev) * np.cos(azim) * r)
    y = np.array(np.sin(elev) * np.sin(azim) * r)
    z = np.array(np.cos(elev) * r)
    return x, y, z


def cart_to_sph_3d(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray):
    r = np.sqrt(x**2 + y**2 + z**2)
    safe_r = np.where(r == 0, 1e-10, r)
    theta = np.arccos(z / safe_r)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2*np.pi)
    theta = np.where(r == 0, 0, theta)
    phi = np.where(r == 0, 0, phi)
    return r, theta, phi


def cart_to_sph_2d(x: float | np.ndarray, y: float | np.ndarray):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2*np.pi)
    phi = np.where(r == 0, 0, phi)
    return r, phi