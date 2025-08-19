from abc import ABC, abstractmethod
import numpy as np


class CWaveModel(ABC):
    def __init__(self):
        self.type = None

    @abstractmethod
    def p(self, t: float | np.ndarray, pos_xyz: tuple):
        pass

    @abstractmethod
    def vec(self, ref_point: tuple | np.ndarray):
        pass


class CWaveModelPlanar(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, azim: float=0, elev: float=0) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.azim = azim
        self.elev = elev
        self.type = "planar"
        self.omega = 2*np.pi*f
        self.k = self.omega / self.c
        self.kx = self.k * np.cos(self.azim) * np.cos(self.elev)
        self.ky = self.k * np.sin(self.azim) * np.cos(self.elev)
        self.kz = self.k * np.sin(self.elev)
    
    def vec(self, ref_point: tuple | np.ndarray = None):
        vec = np.array([
                np.cos(self.azim) * np.cos(self.elev),
                np.sin(self.azim) * np.cos(self.elev),
                np.sin(self.elev)
            ])
        return vec

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if len(t.shape) > 0 and pos_xyz.shape != (3,):
            raise ValueError("Either fix one point in time or one in space!")
        phase = self.kx * pos_xyz[0] + self.ky * pos_xyz[1] + self.kz * pos_xyz[2] - self.omega * t
        return self.amp * np.exp(1j * phase)
    

class CWaveModelSpheric(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, source_xyz: tuple=(0, 0, 0)) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.source_xyz = source_xyz
        self.type = "spheric"
        self.omega = 2*np.pi*f
        self.k = self.omega / self.c

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if t.ndim > 0 and pos_xyz.ndim > 1 and pos_xyz.shape[0] > 3:
            raise ValueError("Either fix one point in time or one in space!")
        
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

    def vec(self, ref_point: tuple | np.ndarray):
        delta_vec = np.asarray(ref_point) - np.asarray(self.source_xyz)
        vec = delta_vec / np.linalg.norm(delta_vec)
        return vec
