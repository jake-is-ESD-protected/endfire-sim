import numpy as np
from .wave import CWaveModel, CWaveModelPlanar
from .algo import sph_to_cart_3d, sph_to_cart_2d
from math import comb

class CSensor:
    def __init__(self, xyz: tuple):
        self.xyz = xyz
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        gain = 1.
        return wave_model.p(t, self.xyz), gain
    
    def direction_vec(self):
        return (0, 0, 0)
    
    def get_sph_mesh(self, n_gridpoints):
        elev = np.linspace(0, np.pi, n_gridpoints)
        azim = np.linspace(0, 2*np.pi, n_gridpoints)
        elev_grid, azim_grid = np.meshgrid(elev, azim)
        return elev_grid, azim_grid
    
    def gain_2d(self, at_freq=None):
        azim = np.linspace(0, 2*np.pi, 100)
        unity_gain = np.ones_like(azim)
        x, y = sph_to_cart_2d(unity_gain, azim)
        return unity_gain, (x, y)
    
    def gain_3d(self, at_freq=None):
        elev_grid, azim_grid = self.get_sph_mesh(30)
        x, y, z = sph_to_cart_3d(np.ones_like(elev_grid), elev_grid, azim_grid)
        return np.ones_like(x), (x, y, z)
    
    def to_plot(self, ax, size=0.5, color='k', log=False, at_freq=None):
        vec = self.direction_vec()
        is_3d = hasattr(ax, 'zaxis')
        
        if is_3d:
            # ===== 3D VERSION =====
            gain, xyz = self.gain_3d(at_freq=at_freq)
            r = np.clip(gain, 1e-7, None)
            r = 20*np.log10(r / np.max(r))
            norm = np.max(np.abs(r))
            r = size * (1 + r / (norm if norm else 1.))
            
            X = self.xyz[0] + r * xyz[0]
            Y = self.xyz[1] + r * xyz[1]
            Z = self.xyz[2] + r * xyz[2]
            
            ax.plot_surface(X, Y, Z, color=color, alpha=0.2, edgecolor='k', linewidth=0.5, label="Cardioid")
            
            arrow_len = size * 2.5
            ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                    arrow_len*vec[0], arrow_len*vec[1], arrow_len*vec[2],
                    color=color, arrow_length_ratio=0.1, linewidth=2)
            ax.scatter(*self.xyz, color='blue', s=50)
            
        else:
            # ===== 2D VERSION =====
            gain, xy = self.gain_2d(at_freq=at_freq)
            gain = np.clip(gain, 1e-7, None)
            if log:
                r = size * (1 + 20*np.log10(gain/np.max(gain)) / 50)
            else:
                r = size * gain / np.max(gain)
            
            X = self.xyz[0] + r * xy[0]
            Y = self.xyz[1] + r * xy[1]
            
            ax.plot(X, Y, color, linestyle='-', linewidth=2)
            
            arrow_len = size * 1.5
            if np.sqrt(vec[0]**2 + vec[1]**2) > 0.5:
                ax.arrow(self.xyz[0], self.xyz[1],
                        arrow_len*vec[0], arrow_len*vec[1],
                        width=0.05, head_width=0.15, head_length=0.2,
                        fc=color, ec='k')
            ax.plot(self.xyz[0], self.xyz[1], 'bo', markersize=5)
            ax.set_aspect('equal')
        

class CCardioidIdeal(CSensor):
    def __init__(self, xyz: tuple, elev: float, azim: float):
        super().__init__(xyz=xyz)
        self.elev = elev
        self.azim = azim
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        p = wave_model.p(t, self.xyz)
        wave_vec = wave_model.vec(self.xyz)
        cardioid_vec = np.array(self.direction_vec())
        gain = np.sqrt(((1 + np.dot(-cardioid_vec, wave_vec)) * 2))
        return p * gain, gain
    
    def direction_vec(self):
        return sph_to_cart_3d(1, self.elev, self.azim)
    
    def gain_2d(self, at_freq=None):
        azim = np.linspace(0, 2*np.pi, 100)
        azim_delta = azim - self.azim
        azim_delta = (azim_delta + np.pi) % (2 * np.pi) - np.pi # avoids precision issues
        gain = 1 + np.cos(azim_delta)
        x, y = sph_to_cart_2d(1, azim)
        return gain, (x, y)
    
    def gain_3d(self, at_freq=None):
        elev_grid, azim_grid = self.get_sph_mesh(30)
        x, y, z = sph_to_cart_3d(np.ones_like(elev_grid), elev_grid, azim_grid)
        vec = self.direction_vec()
        dot_product = vec[0]*x + vec[1]*y + vec[2]*z
        return 1 + dot_product, (x, y, z)


class CEndfire(CSensor):
    def __init__(self, xyz: tuple, distance=None, target_freq=None, n_sensors=2,
                 elev=None, azim=None, c=343.0):
        super().__init__(xyz)
        self.n_sensors = n_sensors
        self.elev = elev
        self.azim = azim
        self.c = c
        direction_vec = np.array(self.direction_vec())
        
        if distance is not None and target_freq is not None:
            raise ValueError("Provide either distance or target_freq, not both.")
        elif target_freq is not None:
            self.freq = target_freq
            self.distance = c / (4 * target_freq)
        elif distance is not None:
            self.distance = distance
            self.freq = c / (4 * distance)
        else:
            raise ValueError("Specify either distance or target_freq.")
        
        sensor_positions = []
        for i in range(n_sensors):
            pos = tuple(np.array(self.xyz) + direction_vec * self.distance * i)
            sensor_positions.append(pos)
        self.poss = tuple(sensor_positions)
        self.path_delay = self.distance / self.c
    
    def direction_vec(self):
        return sph_to_cart_3d(1, self.elev, self.azim)
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray, order: int = None):
        ps = []
        for i, pos in enumerate(self.poss):
            monop = CSensor(pos)
            p, _ = monop.receive(wave_model, t - self.path_delay * i)
            ps.append(p)
        ps = np.vstack(ps)
        self.ps = ps

        if order:
            # differential BF
            n_d = self.path_delay * 48000
            num_mics = len(ps)

            if order >= num_mics:
                raise ValueError(f"Order {order} requires {order+1} mics. Have {num_mics}.")

            p_tot = np.zeros_like(ps[0])
            for m in range(order + 1):
                coefficient = (-1)**m * comb(order, m) # Calculate binomial coefficient
                delayed_signal = np.roll(ps[m], -m * n_d)
                p_tot += coefficient * delayed_signal
        else:
            # DAS BF
            p_tot = np.sum(ps, axis=0)

        gain = np.sqrt(np.mean(np.abs(p_tot)**2) / np.mean(np.abs(ps[0])**2))
        return p_tot, gain
    
    def gain_2d(self, at_freq=None):
        azim_grid = np.linspace(0, 2*np.pi, 100)
        unity_gain = np.ones_like(azim_grid)
        x, y = sph_to_cart_2d(unity_gain, azim_grid)
        gains = np.zeros_like(azim_grid)
        f = at_freq if at_freq else self.freq
        for i, azim in enumerate(azim_grid):
            pw = CWaveModelPlanar(f, azim=azim+np.pi)
            _, gain = self.receive(pw, 0)
            gains[i] = gain
        return gains, (x, y)
    
    def gain_3d(self, at_freq=None):
        elev_grid, azim_grid = self.get_sph_mesh(30)
        x, y, z = sph_to_cart_3d(np.ones_like(elev_grid), elev_grid, azim_grid)
        gains = np.zeros_like(elev_grid)
        for i in range(elev_grid.shape[0]):
            for j in range(elev_grid.shape[1]):
                azim = azim_grid[i,j] + np.pi
                elev = elev_grid[i,j]
                pw = CWaveModelPlanar(self.freq, azim=azim, elev=elev)
                _, gain = self.receive(pw, 0)
                gains[i,j] = gain
        return gains, (x, y, z)
    
    def to_plot(self, ax, size=0.5, color='k', log=False, at_freq=None):
        super().to_plot(ax, size, color, log, at_freq)
        is_3d = hasattr(ax, 'zaxis')

        if is_3d:
            for pos in self.poss[1:]:
                ax.scatter(*tuple(pos), color='blue', s=50)
        else:
            for pos in self.poss[1:]:
                ax.plot(pos[0], pos[1], 'bo', markersize=5)


class CCardioidSynthetic(CEndfire):
    def __init__(self, xyz: tuple, distance=None, target_freq=None, n_sensors=2,
                 elev=None, azim=None, c=343.0):
        super().__init__(xyz=xyz, distance=distance, target_freq=target_freq, n_sensors=n_sensors,
                         elev=elev, azim=azim, c=c)
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray, at_freq=None):
        p = wave_model.p(t, self.xyz)
        wave_vec = wave_model.vec(self.xyz)
        cardioid_vec = np.array(self.direction_vec())
        f = at_freq if at_freq else self.freq
        gain = self.gain_function(f, dot=np.dot(cardioid_vec, wave_vec))
        return p * gain, gain
    
    def gain_function(self, f, th=None, dot=None):
        if not th and not dot:
            raise ValueError("Provide either an angle or dot product!")
        omega = 2*np.pi*f
        k = omega / self.c
        if not dot:
            dot = np.cos(th + np.pi)
        return 2*np.abs(np.cos(k*self.distance/2*dot + omega * self.path_delay/2))
    
    def gain_2d(self, at_freq=None):
        azim = np.linspace(0, 2*np.pi, 100)
        azim_delta = azim - self.azim
        azim_delta = (azim_delta + np.pi) % (2 * np.pi) - np.pi # avoids precision issues
        f = at_freq if at_freq else self.freq
        gain = self.gain_function(f, th=azim_delta + np.pi)
        x, y = sph_to_cart_2d(1, azim)
        return gain, (x, y)
    
    def gain_3d(self, at_freq=None):
        elev_grid, azim_grid = self.get_sph_mesh(30)
        x, y, z = sph_to_cart_3d(np.ones_like(elev_grid), elev_grid, azim_grid)
        vec = self.direction_vec()
        dot_product = vec[0]*x + vec[1]*y + vec[2]*z
        f = at_freq if at_freq else self.freq
        gain = self.gain_function(f, dot=-dot_product)
        return gain, (x, y, z)