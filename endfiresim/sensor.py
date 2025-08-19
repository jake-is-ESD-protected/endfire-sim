import numpy as np
from .wave import CWaveModel, CWaveModelPlanar

class CSensor:
    def __init__(self, xyz: tuple):
        self.xyz = xyz
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        gain = 1.
        return wave_model.p(t, self.xyz), gain


class CCardioidIdeal(CSensor):
    def __init__(self, xyz: tuple, azim: float, elev: float):
        super().__init__(xyz=xyz)
        self.azim = azim
        self.elev = elev
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        p = wave_model.p(t, self.xyz)
    
        if wave_model.type == "planar":
            wave_vec = np.array([
                np.cos(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.elev)
            ])
        elif wave_model.type == "spheric":
            delta_vec = np.asarray(self.xyz) - np.asarray(wave_model.source_xyz)
            wave_vec = delta_vec / np.linalg.norm(delta_vec)
        else:
            raise ValueError(f"Unknown wave model type: {wave_model.type}")

        cardioid_vec = self.__vec()
        gain = np.sqrt(((1 + np.dot(-cardioid_vec, wave_vec)) * 2))
        return p * gain, gain
    
    def __vec(self):
        return np.array([
            np.cos(self.azim) * np.cos(self.elev),
            np.sin(self.azim) * np.cos(self.elev),
            np.sin(self.elev)])
    
    def toPlot(self, ax, size=0.5):
        vec = self.__vec()
        azim = np.arctan2(vec[1], vec[0])
        elev = np.arcsin(vec[2])
        
        is_3d = hasattr(ax, 'zaxis')
        
        if is_3d:
            # ===== 3D VERSION =====
            theta = np.linspace(0, np.pi, 30)
            phi = np.linspace(0, 2*np.pi, 30)
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            x = np.sin(theta_grid) * np.cos(phi_grid)
            y = np.sin(theta_grid) * np.sin(phi_grid)
            z = np.cos(theta_grid)
            
            dot_product = vec[0]*x + vec[1]*y + vec[2]*z
            
            r = 1 + dot_product
            r = np.clip(r, 1e-7, None)
            r = 20*np.log10(r / np.max(r))
            r = size * (1 + r / np.max(np.abs(r)))
            
            X = self.xyz[0] + r * x
            Y = self.xyz[1] + r * y
            Z = self.xyz[2] + r * z
            
            ax.plot_surface(X, Y, Z, color='red', alpha=0.2, edgecolor='k', linewidth=0.5, label="Cardioid")
            
            arrow_len = size * 2.5
            ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                    arrow_len*vec[0], arrow_len*vec[1], arrow_len*vec[2],
                    color='red', arrow_length_ratio=0.1, linewidth=2)
            
        else:
            # ===== 2D VERSION =====
            theta = np.linspace(0, 2*np.pi, 100)
            r = size * (1 + np.cos(theta - azim))
            r = np.clip(r, 1e-7, None)
            r = 20*np.log10(r / np.max(r))
            r = size * (1 + r / np.max(np.abs(r)))
            
            x_card = self.xyz[0] + r * np.cos(theta)
            y_card = self.xyz[1] + r * np.sin(theta)
            
            ax.plot(x_card, y_card, 'k-', linewidth=2)
            
            arrow_len = size * 1.5
            ax.arrow(self.xyz[0], self.xyz[1],
                    arrow_len*np.cos(azim), arrow_len*np.sin(azim),
                    width=0.05, head_width=0.15, head_length=0.2,
                    fc='red', ec='k')
        
        if is_3d:
            ax.scatter(*self.xyz, color='blue', s=50)
        else:
            ax.plot(self.xyz[0], self.xyz[1], 'bo', markersize=5)
        
        if not is_3d:
            ax.set_aspect('equal')


class CCardioidEndfire(CCardioidIdeal):
    def __init__(self, xyz=None, distance=None, target_freq=None, positions=None, 
                 azim=None, elev=None, c=343.0):
        self.c = c
        
        if positions is not None:
            self.xyz1, self.xyz2 = np.asarray(positions[0]), np.asarray(positions[1])
            self.distance = np.linalg.norm(self.xyz2 - self.xyz1)
            delta = self.xyz2 - self.xyz1
            self.azim = np.arctan2(delta[1], delta[0]) if azim is None else azim
            self.elev = np.arcsin(delta[2] / self.distance) if elev is None else elev
            self.freq = c / (4 * self.distance)
            super().__init__(xyz=self.xyz1, azim=self.azim, elev=self.elev)
        else:
            if xyz is None or azim is None or elev is None:
                raise ValueError("For single-position mode, specify xyz, azim, and elev.")
            
            self.xyz1 = np.asarray(xyz)
            self.azim, self.elev = azim, elev
            super().__init__(xyz=self.xyz1, azim=self.azim, elev=self.elev)
            
            direction_vec = self._CCardioidIdeal__vec()
            
            if distance is not None and target_freq is not None:
                raise ValueError("Provide either distance or target_freq, not both.")
            elif target_freq is not None:
                self.freq = target_freq
                self.distance = c / (4 * target_freq)
                self.xyz2 = self.xyz1 + direction_vec * self.distance
            elif distance is not None:
                self.distance = distance
                self.freq = c / (4 * distance)
                self.xyz2 = self.xyz1 + direction_vec * distance
            else:
                raise ValueError("Specify either distance or target_freq.")

        self.synthetic_delay = 1 / (4 * self.freq)

    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        if wave_model.type == "planar":
            wave_vec = np.array([
                np.cos(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.elev)
            ])
        elif wave_model.type == "spheric":
            delta_vec = np.asarray(self.xyz) - np.asarray(wave_model.source_xyz)
            wave_vec = delta_vec / np.linalg.norm(delta_vec)

        array_axis = self._CCardioidIdeal__vec()
        cos_theta = np.dot(wave_vec, -array_axis)
        natural_delay = (self.distance * cos_theta) / self.c

        total_delay = self.synthetic_delay - natural_delay

        p1, _ = CSensor.receive(self, wave_model, t)
        p2, _ = CSensor.receive(self, wave_model, t - total_delay)
        p = p1 + p2
        gain = np.sqrt(np.max(np.abs(p**2)) / np.max(np.abs(p1**2)))
        return p, gain

    def toPlot(self, ax, size=0.5):
        vec = self._CCardioidIdeal__vec()
        azim = np.arctan2(vec[1], vec[0])
        elev = np.arcsin(vec[2])
        
        is_3d = hasattr(ax, 'zaxis')
        
        if is_3d:
            # ===== 3D VERSION =====
            theta = np.linspace(0, np.pi, 30)
            phi = np.linspace(0, 2*np.pi, 30)
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            x = np.sin(theta_grid) * np.cos(phi_grid)
            y = np.sin(theta_grid) * np.sin(phi_grid)
            z = np.cos(theta_grid)
            
            gains = np.zeros_like(theta_grid)
            for i in range(theta_grid.shape[0]):
                for j in range(theta_grid.shape[1]):
                    azim = phi_grid[i,j] + np.pi
                    elev = np.pi/2 - theta_grid[i,j]
                    pw = CWaveModelPlanar(self.freq, azim=azim, elev=elev)
                    _, gain = self.receive(pw, 0)
                    gains[i,j] = gain
            r = np.clip(gains, 1e-7, None)
            r = 20*np.log10(r / np.max(r))
            r = size * (1 + r / np.max(np.abs(r)))
            
            X = self.xyz[0] + r * x
            Y = self.xyz[1] + r * y
            Z = self.xyz[2] + r * z
            
            ax.plot_surface(X, Y, Z, color='red', alpha=0.2, edgecolor='k', linewidth=0.5, label="Cardioid")
            
            arrow_len = size * 2.5
            ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                    arrow_len*vec[0], arrow_len*vec[1], arrow_len*vec[2],
                    color='red', arrow_length_ratio=0.1, linewidth=2)
            
        else:
            # ===== 2D VERSION =====
            theta = np.linspace(0, 2*np.pi, 100)
            gains = np.array([])
            for th in theta:
                pw = CWaveModelPlanar(self.freq, azim=th+np.pi, elev=self.elev)
                p, gain = self.receive(pw, 0)
                gains = np.append(gains, gain)
            r = np.clip(gains, 1e-7, None)
            r = 20*np.log10(r / np.max(r))
            r = size * (1 + r / np.max(np.abs(r)))
            
            x_card = self.xyz[0] + r * np.cos(theta)
            y_card = self.xyz[1] + r * np.sin(theta)
            
            ax.plot(x_card, y_card, 'k-', linewidth=2)
            
            arrow_len = size * 1.5
            ax.arrow(self.xyz[0], self.xyz[1],
                    arrow_len*np.cos(azim), arrow_len*np.sin(azim),
                    width=0.05, head_width=0.15, head_length=0.2,
                    fc='red', ec='k')
        
        if is_3d:
            ax.scatter(*self.xyz1, color='blue', s=50)
            ax.scatter(*self.xyz2, color='blue', s=50)
        else:
            ax.plot(self.xyz1[0], self.xyz1[1], 'bo', markersize=5)
            ax.plot(self.xyz2[0], self.xyz2[1], 'bo', markersize=5)
        
        if not is_3d:
            ax.set_aspect('equal')