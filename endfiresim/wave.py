from abc import ABC, abstractmethod
import numpy as np
from .algo import sph_to_cart_3d


class CWaveModel(ABC):
    def __init__(self):
        self.type = None

    @abstractmethod
    def p(self, t: float | np.ndarray, pos_xyz: tuple):
        pass

    @abstractmethod
    def vec(self, ref_point: tuple | np.ndarray):
        pass

    def __add__(self, other):
        if isinstance(other, CWaveModel):
            return CWaveSuperposition(self, other)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)


class CWaveSuperposition(CWaveModel):
    def __init__(self, *waves):
        self.waves = list(waves)
        self.type = "superposition"
    
    def __add__(self, other):
        if isinstance(other, CWaveModel):
            if isinstance(other, CWaveSuperposition):
                return CWaveSuperposition(*(self.waves + other.waves))
            else:
                return CWaveSuperposition(*(self.waves + [other]))
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray):
        total_p = sum(wave.p(t, pos_xyz) for wave in self.waves)
        return total_p
    
    def vec(self, ref_point: tuple | np.ndarray = None):
        return [wave.vec(ref_point) for wave in self.waves]


class CWaveModelPlanar(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, elev: float=np.pi/2, azim: float=0) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.elev = elev
        self.azim = azim
        self.type = "planar"
        self.omega = 2*np.pi*f
        self.k = self.omega / self.c
        x, y, z = sph_to_cart_3d(np.ones_like(self.azim), self.elev, self.azim)
        self.kx = self.k * x
        self.ky = self.k * y
        self.kz = self.k * z
    
    def vec(self, ref_point: tuple | np.ndarray = None):
        return sph_to_cart_3d(1, self.elev, self.azim)

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if len(t.shape) > 0 and pos_xyz.shape != (3,):
            raise ValueError("Either fix one point in time or one in space!")
        phase = self.kx * pos_xyz[0] + self.ky * pos_xyz[1] + self.kz * pos_xyz[2] - self.omega * t
        return self.amp * np.exp(1j * phase)
    

class CWaveModelSpheric(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, source_xyz: tuple=(0, 0, 0), fs: int=8000, duration: float=1.0) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.source_xyz = source_xyz
        self.type = "spheric"
        self.fs = fs
        self.duration = duration
        if f != "wn":
            self.omega = 2*np.pi*f
            self.k = self.omega / self.c
            self.noise_series = None
        else:
            self.omega = None
            self.k = None
            n_samples = int(self.fs * self.duration)
            self.noise_series = np.random.randn(n_samples) + 1j*np.random.randn(n_samples)

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if t.ndim > 0 and pos_xyz.ndim > 1 and pos_xyz.shape[0] > 3:
            raise ValueError("Either fix one point in time or one in space!")
        
        if self.f == "wn":
            delta = pos_xyz - np.reshape(self.source_xyz, (3,) + (1,)*(pos_xyz.ndim-1))
            r = np.linalg.norm(delta, axis=0)
            time_delay = r / self.c
            delayed_t = t - time_delay
            delayed_idx = (delayed_t * self.fs).astype(int)
            valid = (delayed_idx >= 0) & (delayed_idx < len(self.noise_series))
            values = np.zeros_like(delayed_idx, dtype=complex)
            values[valid] = self.noise_series[delayed_idx[valid]]
            with np.errstate(divide='ignore', invalid='ignore'):
                magnitude = np.where(r > 0.1, self.amp / r, np.inf)
                wave = magnitude * values
            result = np.where(np.isinf(magnitude), np.nan, wave)
            return np.real_if_close(np.squeeze(result))
        
        delta = pos_xyz - np.reshape(self.source_xyz, (3,) + (1,)*(pos_xyz.ndim-1))
        r = np.linalg.norm(delta, axis=0)
        if t.ndim == 0:
            full_phase = self.k * r - self.omega * t
        else:
            full_phase = self.k * r - self.omega * t[..., np.newaxis, np.newaxis]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitude = np.where(r > 1/self.k, self.amp / r, np.inf)
            wave = magnitude * np.exp(-1j * full_phase)
        result = np.where(np.isinf(magnitude), np.nan, wave)
        return np.real_if_close(np.squeeze(result))

    def vec(self, ref_point: tuple | np.ndarray = None):
        if ref_point is None:
            return np.array([1.0, 0.0, 0.0])
        delta_vec = np.asarray(ref_point) - np.asarray(self.source_xyz)
        vec = delta_vec / np.linalg.norm(delta_vec)
        return vec
