"""Block definitions for continuous and discrete Dynamic Systems modeling."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
from CORE.common_math import parse_eng_unit


class Block:
    """Base class for dynamic simulation blocks."""

    def __init__(self, name: str, num_inputs: int = 1, num_outputs: int = 1):
        self.name = name.strip()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = [0.0] * num_inputs
        self.outputs = [0.0] * num_outputs
        # State vector
        self.states = np.zeros(0, dtype=float)
        self.direct_feedthrough: bool = True  # True if output depends directly on input at time t

    @property
    def num_states(self) -> int:
        return len(self.states)

    def compute_output(self, t: float) -> List[float]:
        """Calculates outputs given current states and inputs."""
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        """Calculates time derivative d(states)/dt."""
        return np.zeros(0, dtype=float)

    def reset_states(self) -> None:
        pass


class IntegratorBlock(Block):
    """Integrator: x_dot = u, y = x."""

    def __init__(self, name: str, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.initial_condition = float(initial_condition)
        self.states = np.array([self.initial_condition], dtype=float)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.states[0])
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        return np.array([self.inputs[0]], dtype=float)

    def reset_states(self) -> None:
        self.states = np.array([self.initial_condition], dtype=float)


class GainBlock(Block):
    """Proportional Gain: y = K * u."""

    def __init__(self, name: str, gain: Union[str, float] = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.gain = parse_eng_unit(gain)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.gain * self.inputs[0]
        return self.outputs


class SumBlock(Block):
    """Sum / Difference Junction: y = signs[0]*u[0] + signs[1]*u[1] + ..."""

    def __init__(self, name: str, signs: str = "+-"):
        super().__init__(name, num_inputs=len(signs), num_outputs=1)
        self.signs = [1.0 if s == "+" else -1.0 for s in signs]
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        total = 0.0
        for i, s in enumerate(self.signs):
            if i < len(self.inputs):
                total += s * self.inputs[i]
        self.outputs[0] = total
        return self.outputs


class PIDBlock(Block):
    """Continuous PID Controller: u_out = Kp*e + Ki*integral(e) + Kd*de/dt."""

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        n_filter: float = 100.0  # Filter coefficient for derivative
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n_filter = float(n_filter)
        # States: [x_integral, x_derivative_filter]
        self.states = np.zeros(2, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        e = self.inputs[0]
        x_i, x_d = self.states[0], self.states[1]
        p_term = self.kp * e
        i_term = self.ki * x_i
        d_term = self.kd * self.n_filter * (e - x_d)
        self.outputs[0] = p_term + i_term + d_term
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        e = self.inputs[0]
        x_d = self.states[1]
        dx_i = e
        dx_d = self.n_filter * (e - x_d)
        return np.array([dx_i, dx_d], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(2, dtype=float)


class TransferFunctionBlock(Block):
    """LTI Transfer Function Block H(s) represented in Controllable Canonical State-Space."""

    def __init__(self, name: str, num: Sequence[float], den: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        num_arr = np.asarray(num, dtype=float)
        den_arr = np.asarray(den, dtype=float)
        
        # Convert transfer function to state space: x_dot = A x + B u, y = C x + D u
        tf_sys = signal.TransferFunction(num_arr, den_arr)
        ss_sys = tf_sys.to_ss()
        
        self.A = ss_sys.A
        self.B = ss_sys.B
        self.C = ss_sys.C
        self.D = ss_sys.D

        self.states = np.zeros(self.A.shape[0], dtype=float)
        self.direct_feedthrough = bool(np.abs(self.D[0, 0]) > 1e-12)

    def compute_output(self, t: float) -> List[float]:
        u = np.array([[self.inputs[0]]], dtype=float)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs[0] = float(y[0, 0])
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array([[self.inputs[0]]], dtype=float)
        x = self.states.reshape(-1, 1)
        x_dot = self.A @ x + self.B @ u
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.zeros(self.A.shape[0], dtype=float)


class StepSourceBlock(Block):
    """Step Input Source: u(t) = amplitude for t >= step_time else 0."""

    def __init__(self, name: str, step_time: float = 0.0, amplitude: float = 1.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.step_time = float(step_time)
        self.amplitude = float(amplitude)
        self.initial_value = float(initial_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.amplitude if t >= self.step_time else self.initial_value
        return self.outputs


class SineSourceBlock(Block):
    """Sine Input Source: u(t) = offset + amplitude * sin(2*pi*freq*t + phase)."""

    def __init__(self, name: str, freq: float = 1.0, amplitude: float = 1.0, offset: float = 0.0, phase: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.offset = float(offset)
        self.phase = float(phase)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.offset + self.amplitude * math.sin(2.0 * math.pi * self.freq * t + math.radians(self.phase))
        return self.outputs


class SaturationBlock(Block):
    """Limits signal within [lower_limit, upper_limit]."""

    def __init__(self, name: str, lower_limit: float = -1.0, upper_limit: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        self.outputs[0] = max(self.lower_limit, min(u, self.upper_limit))
        return self.outputs


class ScopeSinkBlock(Block):
    """Records signal history for plotting and analysis."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.time_history: List[float] = []
        self.signal_history: List[float] = []
        self.direct_feedthrough = False

    def record(self, t: float) -> None:
        self.time_history.append(t)
        self.signal_history.append(self.inputs[0])

    def reset_states(self) -> None:
        self.time_history.clear()
        self.signal_history.clear()
