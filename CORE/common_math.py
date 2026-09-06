"""Common mathematical utilities, engineering unit parsing, and waveform structures."""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

# Standard engineering unit multipliers
# SPICE standard rules: MEG is 1e6, M is milli (1e-3) or mega depending on context,
# but we support unambiguous standard engineering notations:
# T=1e12, G=1e9, MEG=1e6, M (in Hz/Ohm context)=1e6, k/K=1e3, m=1e-3, u/µ=1e-6, n=1e-9, p=1e-12, f=1e-15
ENG_PREFIXES: Dict[str, float] = {
    't': 1e12,
    'tera': 1e12,
    'g': 1e9,
    'giga': 1e9,
    'meg': 1e6,
    'mega': 1e6,
    'mhz': 1e6,
    'megohm': 1e6,
    'k': 1e3,
    'kilo': 1e3,
    'khz': 1e3,
    'm': 1e-3,
    'milli': 1e-3,
    'mv': 1e-3,
    'ma': 1e-3,
    'ms': 1e-3,
    'u': 1e-6,
    'µ': 1e-6,
    'micro': 1e-6,
    'us': 1e-6,
    'uv': 1e-6,
    'ua': 1e-6,
    'uf': 1e-6,
    'uh': 1e-6,
    'n': 1e-9,
    'nano': 1e-9,
    'ns': 1e-9,
    'nf': 1e-9,
    'nh': 1e-9,
    'p': 1e-12,
    'pico': 1e-12,
    'ps': 1e-12,
    'pf': 1e-12,
    'f': 1e-15,
    'femto': 1e-15,
    'fs': 1e-15,
}

_UNIT_REGEX = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ°Ω%]*)\s*$'
)


def parse_eng_unit(text: Union[str, int, float]) -> float:
    """Parses engineering strings into floating-point numbers.
    
    Examples:
        '10k' -> 10000.0
        '1u' -> 1e-6
        '100kHz' -> 100000.0
        '2.2MEG' -> 2200000.0
        '5mV' -> 0.005
        '10pF' -> 1e-11
    """
    if isinstance(text, (int, float)):
        return float(text)
    
    text = str(text).strip()
    if not text:
        raise ValueError("Empty string provided to unit parser")
    
    match = _UNIT_REGEX.match(text)
    if not match:
        # Fallback to standard float parse
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"Cannot parse engineering value: '{text}'")
    
    val_str, unit_str = match.groups()
    base_val = float(val_str)
    
    if not unit_str:
        return base_val
    
    u_lower = unit_str.lower()
    
    # Check explicit exact prefixes first
    if u_lower.startswith('meg'):
        return base_val * 1e6
    if u_lower.startswith('g'):
        return base_val * 1e9
    if u_lower.startswith('t'):
        return base_val * 1e12
    if u_lower.startswith('k'):
        return base_val * 1e3
    if u_lower.startswith('u') or u_lower.startswith('µ'):
        return base_val * 1e-6
    if u_lower.startswith('n'):
        return base_val * 1e-9
    if u_lower.startswith('p'):
        return base_val * 1e-12
    if u_lower.startswith('f') and not u_lower.startswith('f') == 'f': # femto vs Farad
        if len(u_lower) > 1 and u_lower[1] in ('s', 'a', 'v', 'd'):
            return base_val * 1e-15
    if u_lower.startswith('f') and u_lower == 'f':
        # Farads, multiplier 1
        return base_val
    if u_lower.startswith('m'):
        # In SPICE 'M' alone or 'm' alone is milli, 'MEG' is mega.
        # But if unit is 'MHz' or 'Mohm', treat as Mega
        if 'hz' in u_lower or 'ohm' in u_lower:
            if unit_str.startswith('M'):
                return base_val * 1e6
        # default m is milli
        return base_val * 1e-3

    return base_val


def format_eng_unit(value: float, unit: str = "", precision: int = 3) -> str:
    """Formats a float into an engineering string with metric prefix."""
    if abs(value) < 1e-18 or math.isnan(value) or math.isinf(value):
        return f"{value:.{precision}g} {unit}".strip()
    
    prefixes = [
        (1e12, 'T'),
        (1e9, 'G'),
        (1e6, 'M'),
        (1e3, 'k'),
        (1.0, ''),
        (1e-3, 'm'),
        (1e-6, 'u'),
        (1e-9, 'n'),
        (1e-12, 'p'),
        (1e-15, 'f'),
    ]
    
    abs_val = abs(value)
    for mult, prefix in prefixes:
        if abs_val >= mult * 0.999:
            scaled = value / mult
            return f"{scaled:.{precision}f} {prefix}{unit}".strip()
    
    return f"{value:.{precision}e} {unit}".strip()


class Waveform:
    """Represents a continuous or discrete signal / time-series / frequency-series."""
    
    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[Union[float, complex]],
        name: str = "signal",
        x_unit: str = "s",
        y_unit: str = "V",
        domain: str = "time"  # 'time', 'frequency', 'discrete'
    ):
        self.name = name
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y)
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.domain = domain
    
    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.y)
    
    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.y)
    
    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.y, deg=True)
    
    @property
    def magnitude_db(self) -> np.ndarray:
        mag = np.abs(self.y)
        # Avoid log(0)
        mag = np.where(mag < 1e-15, 1e-15, mag)
        return 20.0 * np.log10(mag)
    
    def sample_at(self, x_val: float) -> Union[float, complex]:
        """Interpolate value at given x."""
        if len(self.x) == 0:
            return 0.0
        if self.is_complex:
            real_val = float(np.interp(x_val, self.x, np.real(self.y)))
            imag_val = float(np.interp(x_val, self.x, np.imag(self.y)))
            return complex(real_val, imag_val)
        return float(np.interp(x_val, self.x, self.y))
    
    def resample(self, num_points: int) -> Waveform:
        """Resample waveform to uniform grid of num_points."""
        if len(self.x) < 2:
            return self
        new_x = np.linspace(self.x[0], self.x[-1], num_points)
        if self.is_complex:
            new_r = np.interp(new_x, self.x, np.real(self.y))
            new_i = np.interp(new_x, self.x, np.imag(self.y))
            new_y = new_r + 1j * new_i
        else:
            new_y = np.interp(new_x, self.x, self.y)
        return Waveform(new_x, new_y, self.name, self.x_unit, self.y_unit, self.domain)
    
    def compute_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates FFT frequencies and complex spectrum."""
        if len(self.x) < 2 or self.domain != "time":
            return np.array([]), np.array([])
        dt = np.mean(np.diff(self.x))
        if dt <= 0:
            return np.array([]), np.array([])
        n = len(self.x)
        freqs = np.fft.rfftfreq(n, d=dt)
        spectrum = np.fft.rfft(np.real(self.y)) / (n / 2.0)
        return freqs, spectrum
    
    def metrics(self) -> Dict[str, float]:
        """Computes key metrics (min, max, mean, rms, pk-pk, thd)."""
        real_y = np.real(self.y)
        if len(real_y) == 0:
            return {}
        y_min = float(np.min(real_y))
        y_max = float(np.max(real_y))
        y_mean = float(np.mean(real_y))
        y_rms = float(np.sqrt(np.mean(real_y ** 2)))
        y_pkpk = y_max - y_min
        
        res = {
            "min": y_min,
            "max": y_max,
            "mean": y_mean,
            "rms": y_rms,
            "pk_pk": y_pkpk,
        }
        return res
    
    def to_csv(self) -> str:
        """Exports waveform data as CSV text."""
        lines = [f"{self.x_unit},{self.name}_{self.y_unit}"]
        if self.is_complex:
            lines[0] = f"{self.x_unit},{self.name}_mag,{self.name}_phase_deg"
            for x_val, y_val in zip(self.x, self.y):
                lines.append(f"{x_val:.6e},{abs(y_val):.6e},{np.angle(y_val, deg=True):.3f}")
        else:
            for x_val, y_val in zip(self.x, self.y):
                lines.append(f"{x_val:.6e},{float(y_val):.6e}")
        return "\n".join(lines)


class SignalMetrics:
    """Helper methods for electrical and control signal measurements."""
    
    @staticmethod
    def measure_rise_time(
        waveform: Waveform,
        low_pct: float = 0.1,
        high_pct: float = 0.9
    ) -> Optional[float]:
        """Calculates 10%-90% rise time."""
        y = np.real(waveform.y)
        x = waveform.x
        if len(y) < 2:
            return None
        y_min, y_max = np.min(y), np.max(y)
        v_low = y_min + low_pct * (y_max - y_min)
        v_high = y_min + high_pct * (y_max - y_min)
        
        idx_low = np.where(y >= v_low)[0]
        idx_high = np.where(y >= v_high)[0]
        if len(idx_low) == 0 or len(idx_high) == 0:
            return None
        t_low = x[idx_low[0]]
        t_high = x[idx_high[0]]
        return float(abs(t_high - t_low))
    
    @staticmethod
    def measure_cutoff_frequency(
        freqs: np.ndarray,
        mag_db: np.ndarray,
        drop_db: float = -3.0
    ) -> Optional[float]:
        """Finds -3dB cutoff frequency in frequency response."""
        if len(freqs) == 0 or len(mag_db) == 0:
            return None
        passband_gain = mag_db[0]
        target_gain = passband_gain + drop_db
        # Find first frequency where gain drops below target
        idx = np.where(mag_db <= target_gain)[0]
        if len(idx) > 0:
            return float(freqs[idx[0]])
        return None


def split_smart_statements(text: str, delimiter: str = ";") -> List[str]:
    """Splits text on delimiter, ignoring delimiters enclosed inside brackets [], (), or quotes."""
    statements: List[str] = []
    current: List[str] = []
    bracket_depth = 0
    paren_depth = 0
    in_quote: Optional[str] = None

    for char in text:
        if in_quote:
            current.append(char)
            if char == in_quote:
                in_quote = None
        elif char in ('"', "'"):
            in_quote = char
            current.append(char)
        elif char == "[":
            bracket_depth += 1
            current.append(char)
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
        elif char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
        elif char == delimiter and bracket_depth == 0 and paren_depth == 0:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)

    return statements

