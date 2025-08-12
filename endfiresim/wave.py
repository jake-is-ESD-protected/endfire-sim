from abc import ABC, abstractmethod
import numpy as np


class CWaveModel(ABC):
    def __init__(self):
        self.type = None

    @abstractmethod
    def p(self, t: float | np.ndarray, pos_xyz: tuple):
        pass


class CWaveModelPlanar(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, azim: float=0, elev: float=0) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.azim = azim
        self.elev = elev
        self.type = "planar"

    def p(self, t: float | np.ndarray, pos_xyz: tuple):
        omega = 2*np.pi * self.f
        k = omega / self.c
        kx = k * np.cos(self.azim) * np.cos(self.elev)
        ky = k * np.sin(self.azim) * np.cos(self.elev)
        kz = k * np.sin(self.elev)
        phase = kx * pos_xyz[0] + ky * pos_xyz[1] + kz * pos_xyz[2] - omega * t
        return self.amp * np.exp(1j * phase)
    

class CWaveModelSpheric(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, source_xyz: tuple=(0, 0, 0)) -> None:
        raise NotImplementedError()
        self.f = f
        self.amp = amp
        self.c = c
        self.source_xyz = source_xyz
        self.type = "spheric"

    def p(self, t: float | np.ndarray, pos_xyz: np.ndarray):
        pass
