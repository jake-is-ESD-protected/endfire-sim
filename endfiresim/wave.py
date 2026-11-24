from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import hilbert
from .algo import sph_to_cart_3d


def _position_columns(pos_xyz) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=float)
    if pos.ndim == 0 or (pos.ndim == 1 and pos.shape[0] != 3):
        raise ValueError("Positions must contain three Cartesian coordinates.")
    if pos.ndim == 1:
        return pos.reshape(3, 1)
    if pos.shape[0] == 3:
        return pos.reshape(3, -1)
    if pos.shape[-1] == 3:
        return pos.reshape(-1, 3).T
    raise ValueError("Positions must have shape (3, ...) or (..., 3).")


def _pink_noise(n_samples: int) -> np.ndarray:
    white = np.random.randn(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    ramp = np.ones_like(freqs)
    ramp[1:] = 1 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * ramp, n=n_samples)
    return pink / (np.std(pink) + 1e-12)


class CWaveModel(ABC):
    def __init__(self):
        self.type = None

    @abstractmethod
    def p(self, t: float | np.ndarray, pos_xyz: tuple, fs: int | None = None):
        pass

    @abstractmethod
    def vec(self, ref_point: tuple | np.ndarray):
        pass

    def field(self, pos_xyz, n_samples: int, fs: float) -> np.ndarray:
        cols = _position_columns(pos_xyz)
        t = np.arange(n_samples) / fs
        return np.stack([self.p(t, tuple(col), fs) for col in cols.T], axis=1)

    def __add__(self, other):
        if isinstance(other, CWaveModel):
            return CWaveSuperposition(self, other)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)


class CWaveSuperposition(CWaveModel):
    def __init__(self, *waves):
        c = waves[0].c
        if any(wave.c != c for wave in waves[1:]):
            raise ValueError("All waves in a superposition must have the same speed of sound c.")
        self.waves = list(waves)
        self.c = c
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
    
    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray, fs: int | None = None):
        total_p = sum(wave.p(t, pos_xyz, fs=fs) for wave in self.waves)
        return total_p

    def field(self, pos_xyz, n_samples: int, fs: float) -> np.ndarray:
        fields = [wave.field(pos_xyz, n_samples, fs) for wave in self.waves]
        return np.sum(fields, axis=0)

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

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray, fs: int | None = None):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if len(t.shape) > 0 and pos_xyz.shape != (3,):
            raise ValueError("Either fix one point in time or one in space!")
        phase = self.kx * pos_xyz[0] + self.ky * pos_xyz[1] + self.kz * pos_xyz[2] - self.omega * t
        return self.amp * np.exp(1j * phase)

    def field(self, pos_xyz, n_samples: int, fs: float) -> np.ndarray:
        cols = _position_columns(pos_xyz)
        t = np.arange(n_samples) / fs
        phase = (self.kx * cols[0] + self.ky * cols[1] + self.kz * cols[2])[None, :] \
            - self.omega * t[:, None]
        return self.amp * np.exp(1j * phase)  # (n_samples, M)


class CWaveModelSpheric(CWaveModel):
    def __init__(self, f: float, amp: int=1, c=343, source_xyz: tuple=(0, 0, 0), duration: float=1.0) -> None:
        self.f = f
        self.amp = amp
        self.c = c
        self.source_xyz = source_xyz
        self.type = "spheric"
        self.duration = duration
        self._signal_fs = None # caching flag to prevent p() from generating a new signal_series on each call
        if isinstance(f, float) or isinstance(f, int):
            self.omega = 2*np.pi*f
            self.k = self.omega / self.c
            self.signal_series = None
        elif isinstance(f, np.ndarray):
            self.omega = None
            self.k = None
            self.signal_series = hilbert(f)
        elif isinstance(f, str):
            self.omega = None
            self.k = None
            self.signal_series = None
            if f not in ['wn', 'pn']:
                raise ValueError(f"Unknwon signal specifier {f}")
        else:
            raise ValueError(f"Unknwon tpye {f} for source signal.")
            

    def p(self, t: float | np.ndarray, pos_xyz: tuple | np.ndarray, fs: int | None = None):
        t = np.asarray(t)
        pos_xyz = np.asarray(pos_xyz)
        if t.ndim > 0 and pos_xyz.shape != (3,):
            raise ValueError("Either fix one point in time or one in space!")

        if self.omega is None:
            if fs is None:
                raise ValueError("A sampling rate is required for sampled signals.")
            if self.signal_series is None or self._signal_fs != fs:
                n_samples = int(fs * self.duration)
                if isinstance(self.f, str) and self.f == "wn":
                    self.signal_series = np.random.randn(n_samples) + 1j*np.random.randn(n_samples)
                elif isinstance(self.f, str) and self.f == "pn":
                    self.signal_series = _pink_noise(n_samples) + 1j * _pink_noise(n_samples)
                self._signal_fs = fs
            signal_fs = self._signal_fs if isinstance(self.f, str) else fs
            # Field rendering starts at t=0; the absolute time base of t is
            # applied as an integer sample offset afterwards (legacy code
            # floored delayed sample indices the same way).
            n = len(t) if t.ndim > 0 else int(signal_fs * self.duration)
            s = int(float(t[0]) * signal_fs) if t.ndim > 0 else int(float(t) * signal_fs)
            n_gen = n + max(s, 0) + 2
            field = self.field(pos_xyz, n_gen, signal_fs)  # (n_gen, M)
            if t.ndim > 0:
                idx = np.arange(n) + s
                valid = (idx >= 0) & (idx < n_gen)
                result = np.zeros((n, field.shape[1]))
                result[valid] = field[idx[valid]]
                result = result if pos_xyz.ndim > 1 else result[:, 0]
            else:
                result = field[s] if 0 <= s < n_gen else np.full(field.shape[1], np.nan)
                if pos_xyz.ndim == 1:
                    result = result[0]
            return np.squeeze(result)

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

    def field(self, pos_xyz, n_samples: int, fs: float) -> np.ndarray:
        pos_cols = _position_columns(pos_xyz)
        src = np.asarray(self.source_xyz, dtype=float).reshape(3)
        r = np.linalg.norm(pos_cols - src.reshape(3, 1), axis=0)
        if self.omega is not None:
            # monochromatic source: closed-form spherical wave
            t = np.arange(n_samples) / fs
            full_phase = self.k * r[None, :] - self.omega * t[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                magnitude = np.where(r > 1 / self.k, self.amp / r, np.inf)
            out = magnitude[None, :] * np.exp(-1j * full_phase)
            return np.where(np.isinf(magnitude)[None, :], np.nan, out)
        import acoular as ac
        if self.signal_series is None or self._signal_fs != fs:
            n_sig = int(fs * self.duration)
            if isinstance(self.f, str) and self.f == "wn":
                self.signal_series = np.random.randn(n_sig) + 1j * np.random.randn(n_sig)
            elif isinstance(self.f, str) and self.f == "pn":
                self.signal_series = _pink_noise(n_sig) + 1j * _pink_noise(n_sig)
            self._signal_fs = fs
        # Acoular yields blocks, and its source signal determines the number of
        # output samples.  Pad for propagation time so a frame requested at t=0
        # does not lose its tail at the source boundary.
        n_delay = int(np.ceil(np.max(r) / self.c * fs)) + 2
        signal = np.real(self.signal_series)
        if len(signal) < n_samples + n_delay:
            signal = np.pad(signal, (0, n_samples + n_delay - len(signal)))
        mic = ac.MicGeom(pos_total=pos_cols)
        env = ac.Environment(c=self.c)
        ts = ac.TimeSamples(data=signal.reshape(-1, 1), sample_freq=fs)
        gen = ac.GenericSignalGenerator(source=ts)
        source = ac.PointSource(signal=gen, loc=tuple(src), mics=mic, env=env)
        blocks = list(source.result(n_samples))
        out = np.concatenate(blocks, axis=0)[:n_samples] * self.amp
        if out.shape[0] < n_samples:
            out = np.pad(out, ((0, n_samples - out.shape[0]), (0, 0)))
        out[:, r <= 0.1] = np.nan
        return out

    def vec(self, ref_point: tuple | np.ndarray = None):
        if ref_point is None:
            return np.array([1.0, 0.0, 0.0])
        delta_vec = np.asarray(ref_point) - np.asarray(self.source_xyz)
        vec = delta_vec / np.linalg.norm(delta_vec)
        return vec
