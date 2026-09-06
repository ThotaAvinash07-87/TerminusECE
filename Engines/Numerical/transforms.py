"""Fourier, Laplace, and Z-Transforms for continuous and discrete signal analysis."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
from CORE.common_math import Waveform, parse_eng_unit
from CORE.ascii_canvas import AsciiBodePlotter, AsciiPlotter


def fourier_transform(waveform: Waveform) -> Tuple[np.ndarray, np.ndarray]:
    """Computes single-sided FFT amplitude and phase spectrum of a time-domain waveform."""
    return waveform.compute_fft()


def inverse_fourier_transform(spectrum: np.ndarray, dt: float) -> Waveform:
    """Reconstructs time-domain signal from complex spectrum via IFFT."""
    y = np.fft.irfft(spectrum)
    t = np.arange(len(y)) * dt
    return Waveform(t, y, name="ifft_signal", x_unit="s", y_unit="V")


class TransferFunction:
    """Continuous-time Laplace Domain Transfer Function H(s) = Num(s) / Den(s)."""

    def __init__(self, num: Sequence[float], den: Sequence[float], name: str = "H(s)"):
        self.num = np.asarray(num, dtype=float)
        self.den = np.asarray(den, dtype=float)
        self.name = name

        # Normalize so leading denominator coefficient is 1.0
        if len(self.den) == 0 or self.den[0] == 0:
            raise ValueError("Denominator leading coefficient cannot be zero.")
        scale = self.den[0]
        self.num = self.num / scale
        self.den = self.den / scale

        # SciPy LTI system representation
        self.sys = signal.TransferFunction(self.num, self.den)

    @property
    def poles(self) -> np.ndarray:
        return self.sys.poles

    @property
    def zeros(self) -> np.ndarray:
        return self.sys.zeros

    def is_stable(self) -> bool:
        """BIBO stability: all poles must have strictly negative real parts."""
        return bool(np.all(np.real(self.poles) < 0))

    def frequency_response(self, freqs: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates frequency response: returns (freqs, mag_db, phase_deg)."""
        if freqs is None:
            w, mag, phase = signal.bode(self.sys, n=200)
            f = w / (2.0 * math.pi)
            return f, mag, phase
        else:
            w = 2.0 * math.pi * np.asarray(freqs, dtype=float)
            w, mag, phase = signal.bode(self.sys, w=w)
            return np.asarray(freqs), mag, phase

    def step_response(self, t: Optional[Sequence[float]] = None) -> Waveform:
        """Calculates step response y(t) for unit step input u(t)."""
        t_out, y_out = signal.step(self.sys, T=t)
        return Waveform(t_out, y_out, name=f"step({self.name})", x_unit="s", y_unit="Output")

    def impulse_response(self, t: Optional[Sequence[float]] = None) -> Waveform:
        """Calculates impulse response h(t)."""
        t_out, y_out = signal.impulse(self.sys, T=t)
        return Waveform(t_out, y_out, name=f"impulse({self.name})", x_unit="s", y_unit="Output")

    def render_bode_ascii(self, width: int = 70, height: int = 9) -> str:
        """Renders ASCII Bode Plot."""
        f, mag, phase = self.frequency_response()
        return AsciiBodePlotter.plot_bode(f, mag, phase, width=width, height_each=height, title=f"Bode Plot: {self.name}")

    def __repr__(self) -> str:
        num_str = " + ".join([f"{c:.3g}*s^{len(self.num)-1-i}" if len(self.num)-1-i > 0 else f"{c:.3g}" for i, c in enumerate(self.num)])
        den_str = " + ".join([f"{c:.3g}*s^{len(self.den)-1-i}" if len(self.den)-1-i > 0 else f"{c:.3g}" for i, c in enumerate(self.den)])
        return f"{self.name} = ({num_str}) / ({den_str})"


class DiscreteTransferFunction:
    """Discrete-time Z-Domain Transfer Function H(z) = B(z) / A(z)."""

    def __init__(self, b: Sequence[float], a: Sequence[float], dt: float = 1.0, name: str = "H(z)"):
        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.dt = dt
        self.name = name

    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.a)

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.b)

    def is_stable(self) -> bool:
        """Discrete stability: all poles inside unit circle (|z| < 1)."""
        return bool(np.all(np.abs(self.poles) < 1.0))

    def frequency_response(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates digital frequency response up to Nyquist."""
        w, h = signal.freqz(self.b, self.a, worN=num_points)
        freqs = w * (1.0 / (2.0 * math.pi * self.dt))
        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        phase_deg = np.angle(h, deg=True)
        return freqs, mag_db, phase_deg

    def impulse_response(self, num_samples: int = 50) -> Waveform:
        """Calculates discrete unit impulse response."""
        imp = np.zeros(num_samples)
        imp[0] = 1.0
        y = signal.lfilter(self.b, self.a, imp)
        t = np.arange(num_samples) * self.dt
        return Waveform(t, y, name=f"impulse({self.name})", x_unit="s", y_unit="Output", domain="discrete")
