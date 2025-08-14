from abc import ABC, abstractmethod
import numpy as np
from .wave import CWaveModel

class CSensor(ABC):
    def __init__(self, xyz: tuple, azim: float, elev: float):
        self.xyz = xyz
        self.azim = azim
        self.elev = elev
    
    @abstractmethod
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        pass


class CCardioidIdeal(CSensor):
    def __init__(self, xyz: tuple, azim: float, elev: float):
        super().__init__(xyz, azim, elev)
    
    def receive(self, wave_model: CWaveModel, t: float | np.ndarray):
        p = wave_model.p(t, self.xyz)
    
        if wave_model.type == "planar":
            # For planar waves, use the predefined wave direction
            wave_vec = np.array([
                np.cos(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.azim) * np.cos(wave_model.elev),
                np.sin(wave_model.elev)
            ])
        elif wave_model.type == "spheric":
            # For spherical waves, calculate direction from source to receiver
            delta_vec = np.asarray(self.xyz) - np.asarray(wave_model.source_xyz)
            wave_vec = delta_vec / np.linalg.norm(delta_vec)  # Unit vector pointing FROM source TO receiver
        else:
            raise ValueError(f"Unknown wave model type: {wave_model.type}")

        cardioid_vec = self.__vec()  # Unit vector pointing in sensor's direction
        gain = (1 + np.dot(-cardioid_vec, wave_vec)) / 2  # No negation needed now
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
            
            gain = size * (1 + dot_product)
            
            X = self.xyz[0] + gain * x
            Y = self.xyz[1] + gain * y
            Z = self.xyz[2] + gain * z
            
            ax.plot_surface(X, Y, Z, color='red', alpha=0.2, edgecolor='k', linewidth=0.5, label="Cardioid")
            
            arrow_len = size * 2.5
            ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                    arrow_len*vec[0], arrow_len*vec[1], arrow_len*vec[2],
                    color='red', arrow_length_ratio=0.1, linewidth=2)
            
        else:
            # ===== 2D VERSION =====
            theta = np.linspace(0, 2*np.pi, 100)
            r = size * (1 + np.cos(theta - azim))
            
            x_card = self.xyz[0] + r * np.cos(theta)
            y_card = self.xyz[1] + r * np.sin(theta)
            
            ax.plot(x_card, y_card, 'k-', linewidth=2)
            
            arrow_len = size * 1.5
            ax.arrow(self.xyz[0], self.xyz[1],
                    arrow_len*np.cos(azim), arrow_len*np.sin(azim),
                    width=0.05, head_width=0.15, head_length=0.2,
                    fc='red', ec='k')
        
        if is_3d:
            ax.scatter(*self.xyz, color='red', s=50)
        else:
            ax.plot(self.xyz[0], self.xyz[1], 'ro', markersize=5)
        
        if not is_3d:
            ax.set_aspect('equal')


class CCardioidEndfire(CSensor):
    def __init__(self):
        super().__init__()


