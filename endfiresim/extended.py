import numpy as np
from endfiresim.wave import CWaveModel
from .sensor import CEndfire, CSensor

class CBroadside(CEndfire):
    def __init__(self, xyz: tuple, distance=None, target_freq=None, n_sensors=2, elev=None, azim=None, c=343):
        super().__init__(xyz, distance, target_freq, n_sensors, elev, azim, c)
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        ps = []
        for i, pos in enumerate(self.poss):
            monop = CSensor(pos)
            p, _ = monop.receive(wave_model, t)
            ps.append(p)
        ps = np.vstack(ps)
        self.ps = ps
        p_tot = np.sum(ps, axis=0)
        gain = np.sqrt(np.mean(np.abs(p_tot)**2) / np.mean(np.abs(ps[0])**2))
        return p_tot, gain
    
    # def to_plot(self, ax, size=0.5, log=False):
    #     self.xyz
    #     CSensor.to_plot(self, ax, size, log)
        
    #     is_3d = hasattr(ax, 'zaxis')

    #     if is_3d:
    #         for pos in self.poss[1:]:
    #             ax.scatter(*tuple(pos), color='blue', s=50)
    #     else:
    #         for pos in self.poss[1:]:
    #             ax.plot(pos[0], pos[1], 'bo', markersize=5)