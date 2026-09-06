"""Comprehensive block library for continuous, discrete, nonlinear, and physical Dynamic Systems modeling."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
from CORE.common_math import parse_eng_unit, sanitize_array


class Block:
    """Base class for all dynamic simulation blocks."""

    def __init__(self, name: str, num_inputs: int = 1, num_outputs: int = 1):
        self.name = name.strip()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = [0.0] * num_inputs
        self.outputs = [0.0] * num_outputs
        # State vector
        self.states = np.zeros(0, dtype=float)
        self.direct_feedthrough: bool = True  # True if output depends directly on input at time t
        self.sample_time: float = 0.0         # 0.0 for continuous, >0 for discrete

    @property
    def num_states(self) -> int:
        return len(self.states)

    def compute_output(self, t: float) -> List[float]:
        """Calculates outputs given current states, inputs, and time t."""
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        """Calculates time derivative d(states)/dt for ODE integration."""
        return np.zeros(0, dtype=float)

    def discrete_update(self, t: float) -> None:
        """Updates internal discrete states at sample time intervals."""
        pass

    def reset_states(self) -> None:
        """Resets internal states to initial conditions."""
        pass


# =====================================================================
# 1. CONTINUOUS & STATE-SPACE BLOCKS
# =====================================================================

class IntegratorBlock(Block):
    """Continuous Integrator: x_dot = u, y = x.
    Features: initial conditions, saturation limits [lower_limit, upper_limit],
    anti-windup derivative freezing, and external reset triggers.
    """

    def __init__(
        self,
        name: str,
        initial_condition: float = 0.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        reset_type: str = "none"  # 'none', 'rising', 'falling', 'level'
    ):
        num_inputs = 2 if reset_type != "none" else 1
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.initial_condition = float(initial_condition)
        self.lower_limit = float(lower_limit) if lower_limit is not None else None
        self.upper_limit = float(upper_limit) if upper_limit is not None else None
        self.reset_type = reset_type.lower()
        self.prev_reset_in = 0.0

        self.states = np.array([self.initial_condition], dtype=float)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        val = float(self.states[0])
        if self.lower_limit is not None:
            val = max(self.lower_limit, val)
        if self.upper_limit is not None:
            val = min(self.upper_limit, val)
        self.outputs[0] = val
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = self.inputs[0]

        # Handle external reset if configured
        if self.reset_type != "none" and len(self.inputs) > 1:
            rst = self.inputs[1]
            do_reset = False
            if self.reset_type == "level" and abs(rst) > 1e-6:
                do_reset = True
            elif self.reset_type == "rising" and self.prev_reset_in <= 0.0 and rst > 0.0:
                do_reset = True
            elif self.reset_type == "falling" and self.prev_reset_in >= 0.0 and rst < 0.0:
                do_reset = True
            self.prev_reset_in = rst

            if do_reset:
                self.states[0] = self.initial_condition
                return np.array([0.0], dtype=float)

        # Anti-windup clamping: freeze derivative if at limit and driving further into saturation
        curr_x = self.states[0]
        if self.upper_limit is not None and curr_x >= self.upper_limit and u > 0:
            return np.array([0.0], dtype=float)
        if self.lower_limit is not None and curr_x <= self.lower_limit and u < 0:
            return np.array([0.0], dtype=float)

        return np.array([u], dtype=float)

    def reset_states(self) -> None:
        self.states = np.array([self.initial_condition], dtype=float)
        self.prev_reset_in = 0.0


class DerivativeBlock(Block):
    """Filtered Derivative Block: H(s) = s / (tau * s + 1).
    Avoids infinite high-frequency noise amplification of pure derivative s.
    """

    def __init__(self, name: str, tau: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.tau = max(1e-6, float(tau))
        # State: x (filtered state)
        self.states = np.zeros(1, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        x = self.states[0]
        # y = (u - x) / tau
        self.outputs[0] = (u - x) / self.tau
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = self.inputs[0]
        x = self.states[0]
        # dx/dt = (u - x) / tau
        return np.array([(u - x) / self.tau], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(1, dtype=float)


class TransferFunctionBlock(Block):
    """Continuous LTI Transfer Function H(s) = Num(s) / Den(s) in Controllable Canonical State-Space."""

    def __init__(self, name: str, num: Sequence[float], den: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        num_arr = np.asarray(num, dtype=float)
        den_arr = np.asarray(den, dtype=float)

        tf_sys = signal.TransferFunction(num_arr, den_arr)
        ss_sys = tf_sys.to_ss()

        self.A = np.asarray(ss_sys.A, dtype=float)
        self.B = np.asarray(ss_sys.B, dtype=float)
        self.C = np.asarray(ss_sys.C, dtype=float)
        self.D = np.asarray(ss_sys.D, dtype=float)

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


class StateSpaceBlock(Block):
    """Linear State-Space representation: dx/dt = A*x + B*u, y = C*x + D*u."""

    def __init__(
        self,
        name: str,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        x0: Optional[Sequence[float]] = None
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)

        num_in = self.B.shape[1] if self.B.ndim > 1 else 1
        num_out = self.C.shape[0] if self.C.ndim > 1 else 1
        super().__init__(name, num_inputs=num_in, num_outputs=num_out)

        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.zeros(self.A.shape[0], dtype=float)
        self.states = np.array(self.x0, dtype=float)
        self.direct_feedthrough = bool(np.any(np.abs(self.D) > 1e-12))

    def compute_output(self, t: float) -> List[float]:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs = [float(val) for val in y.flatten()]
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        x_dot = self.A @ x + self.B @ u
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.array(self.x0, dtype=float)


class TransportDelayBlock(Block):
    """Pure continuous time delay: y(t) = u(t - delay_time)."""

    def __init__(self, name: str, delay_time: float = 0.1, initial_output: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.delay_time = max(1e-6, float(delay_time))
        self.initial_output = float(initial_output)
        self.buffer_time: List[float] = []
        self.buffer_val: List[float] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        # Record current input with timestamp
        self.buffer_time.append(t)
        self.buffer_val.append(self.inputs[0])

        target_t = t - self.delay_time
        if target_t < 0 or len(self.buffer_time) < 2:
            self.outputs[0] = self.initial_output
        else:
            self.outputs[0] = float(np.interp(target_t, self.buffer_time, self.buffer_val))

        # Trim buffer to save memory
        if len(self.buffer_time) > 500 and target_t > self.buffer_time[100]:
            self.buffer_time = self.buffer_time[100:]
            self.buffer_val = self.buffer_val[100:]

        return self.outputs

    def reset_states(self) -> None:
        self.buffer_time.clear()
        self.buffer_val.clear()


# =====================================================================
# 2. NONLINEAR & ACTUATOR BLOCKS
# =====================================================================

class SaturationBlock(Block):
    """Saturation: clamps input signal to [lower_limit, upper_limit]."""

    def __init__(self, name: str, lower_limit: float = -1.0, upper_limit: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        if self.lower_limit > self.upper_limit:
            self.lower_limit, self.upper_limit = self.upper_limit, self.lower_limit
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        self.outputs[0] = max(self.lower_limit, min(u, self.upper_limit))
        return self.outputs


class RateLimiterBlock(Block):
    """Rate Limiter: limits the time rate of change of the signal:
    falling_slew_rate <= dy/dt <= rising_slew_rate.
    """

    def __init__(
        self,
        name: str,
        rising_slew_rate: float = 100.0,
        falling_slew_rate: float = -100.0,
        initial_output: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.rising_rate = float(rising_slew_rate)
        self.falling_rate = float(falling_slew_rate)
        self.initial_output = float(initial_output) if initial_output is not None else None
        self.prev_t: Optional[float] = None
        self.prev_y: Optional[float] = self.initial_output
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        if self.prev_t is None:
            self.prev_t = t
            if self.prev_y is None:
                self.prev_y = u
            self.outputs[0] = self.prev_y
            return self.outputs

        if t <= self.prev_t:
            self.outputs[0] = self.prev_y if self.prev_y is not None else u
            return self.outputs

        dt = t - self.prev_t
        rate = (u - self.prev_y) / dt
        limited_rate = max(self.falling_rate, min(rate, self.rising_rate))
        y = self.prev_y + limited_rate * dt

        self.prev_t = t
        self.prev_y = y
        self.outputs[0] = y
        return self.outputs

    def reset_states(self) -> None:
        self.prev_t = None
        self.prev_y = self.initial_output


class DeadZoneBlock(Block):
    """Dead Zone: outputs 0 when lower_limit <= u <= upper_limit, else offsets proportionally."""

    def __init__(self, name: str, start_zone: float = -0.5, end_zone: float = 0.5):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.start_zone = float(start_zone)
        self.end_zone = float(end_zone)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        if u > self.end_zone:
            self.outputs[0] = u - self.end_zone
        elif u < self.start_zone:
            self.outputs[0] = u - self.start_zone
        else:
            self.outputs[0] = 0.0
        return self.outputs


class BacklashBlock(Block):
    """Mechanical Backlash / Hysteresis with deadband width 2*deadband_width."""

    def __init__(self, name: str, deadband_width: float = 1.0, initial_output: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.deadband = float(deadband_width)
        self.prev_y = float(initial_output)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        half_band = self.deadband / 2.0
        if u > self.prev_y + half_band:
            self.prev_y = u - half_band
        elif u < self.prev_y - half_band:
            self.prev_y = u + half_band
        self.outputs[0] = self.prev_y
        return self.outputs

    def reset_states(self) -> None:
        self.prev_y = 0.0


class RelayBlock(Block):
    """Relay / Schmitt Trigger (Bang-Bang controller with turn-on/turn-off hysteresis thresholds)."""

    def __init__(
        self,
        name: str,
        switch_on_point: float = 0.5,
        switch_off_point: float = -0.5,
        output_on: float = 1.0,
        output_off: float = 0.0
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.on_thresh = float(switch_on_point)
        self.off_thresh = float(switch_off_point)
        self.y_on = float(output_on)
        self.y_off = float(output_off)
        self.state_on = False
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        if u >= self.on_thresh:
            self.state_on = True
        elif u <= self.off_thresh:
            self.state_on = False
        self.outputs[0] = self.y_on if self.state_on else self.y_off
        return self.outputs

    def reset_states(self) -> None:
        self.state_on = False


class CoulombViscousFrictionBlock(Block):
    """Realistic Mechanical Friction: F = F_coulomb * sgn(v) + b_viscous * v + F_static * exp(-|v|/v_stribeck)."""

    def __init__(
        self,
        name: str,
        f_coulomb: float = 1.0,
        b_viscous: float = 0.1,
        f_static: float = 1.5,
        v_stribeck: float = 0.05
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fc = float(f_coulomb)
        self.b = float(b_viscous)
        self.fs = float(f_static)
        self.vs = max(1e-4, float(v_stribeck))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        v = self.inputs[0]
        stribeck = (self.fs - self.fc) * math.exp(-abs(v) / self.vs)
        friction = (self.fc + stribeck) * np.tanh(v * 50.0) + self.b * v
        self.outputs[0] = float(friction)
        return self.outputs


class QuantizerBlock(Block):
    """Quantization of continuous signal to discrete steps of size q (e.g. ADC/DAC precision)."""

    def __init__(self, name: str, quantization_interval: float = 0.1):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.q = max(1e-9, float(quantization_interval))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        self.outputs[0] = round(u / self.q) * self.q
        return self.outputs


# =====================================================================
# 3. MATH & LOGIC OPERATIONS
# =====================================================================

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


class ProductBlock(Block):
    """Multiplication & Division: y = (u[0] * u[1] ...) / (u[k] ...) with zero-division safeguard."""

    def __init__(self, name: str, operations: str = "**"):
        super().__init__(name, num_inputs=len(operations), num_outputs=1)
        self.ops = operations  # e.g., '*/' for u0 / u1
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        res = 1.0
        for i, op in enumerate(self.ops):
            if i < len(self.inputs):
                val = self.inputs[i]
                if op == "*":
                    res *= val
                elif op == "/":
                    denom = val if abs(val) > 1e-12 else (1e-12 if val >= 0 else -1e-12)
                    res /= denom
        self.outputs[0] = res
        return self.outputs


class MathFunctionBlock(Block):
    """Applies unary math function: sin, cos, tan, exp, log, log10, sqrt, abs, sign, square."""

    FUNC_MAP: Dict[str, Callable[[float], float]] = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": lambda x: math.exp(max(-100.0, min(100.0, x))),
        "log": lambda x: math.log(max(1e-12, x)),
        "log10": lambda x: math.log10(max(1e-12, x)),
        "sqrt": lambda x: math.sqrt(max(0.0, x)),
        "abs": abs,
        "sign": lambda x: float(np.sign(x)),
        "square": lambda x: x * x,
    }

    def __init__(self, name: str, function: str = "sin"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.func_name = function.lower()
        self.func = self.FUNC_MAP.get(self.func_name, math.sin)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        try:
            self.outputs[0] = self.func(u)
        except Exception:
            self.outputs[0] = 0.0
        return self.outputs


class LookupTable1DBlock(Block):
    """1D Interpolated Lookup Table: y = f(u) via linear interpolation."""

    def __init__(self, name: str, x_data: Sequence[float], y_data: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.x_pts = np.asarray(x_data, dtype=float)
        self.y_pts = np.asarray(y_data, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        self.outputs[0] = float(np.interp(u, self.x_pts, self.y_pts))
        return self.outputs


class SwitchBlock(Block):
    """3-Input Switch: outputs input 1 if input 2 >= threshold, else outputs input 3."""

    def __init__(self, name: str, threshold: float = 0.0):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.threshold = float(threshold)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        in1, ctrl, in3 = self.inputs[0], self.inputs[1], self.inputs[2]
        self.outputs[0] = in1 if ctrl >= self.threshold else in3
        return self.outputs


# =====================================================================
# 4. DISCRETE & DIGITAL CONTROL BLOCKS
# =====================================================================

class ZeroOrderHoldBlock(Block):
    """Zero-Order Hold (ZOH): discrete sample-and-hold with sample time Ts."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.held_value = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        # Check if new sample period hit
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.held_value = self.inputs[0]
            self.last_sample_t = sample_instant

        self.outputs[0] = self.held_value
        return self.outputs

    def reset_states(self) -> None:
        self.held_value = 0.0
        self.last_sample_t = -1.0


class UnitDelayBlock(Block):
    """Discrete Unit Delay z^-1: y[k] = u[k-1]."""

    def __init__(self, name: str, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.prev_val = self.curr_val
            self.curr_val = self.inputs[0]
            self.last_sample_t = sample_instant

        self.outputs[0] = self.prev_val
        return self.outputs

    def reset_states(self) -> None:
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_sample_t = -1.0


class PIDBlock(Block):
    """Continuous & Discrete PID Controller with anti-windup clamping and derivative filtering."""

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        n_filter: float = 100.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n_filter = float(n_filter)
        self.lower_limit = float(lower_limit) if lower_limit is not None else None
        self.upper_limit = float(upper_limit) if upper_limit is not None else None

        # States: [x_integral, x_derivative_filter]
        self.states = np.zeros(2, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        e = self.inputs[0]
        x_i, x_d = self.states[0], self.states[1]
        p_term = self.kp * e
        i_term = self.ki * x_i
        d_term = self.kd * self.n_filter * (e - x_d)
        u_raw = p_term + i_term + d_term

        # Output saturation
        u_out = u_raw
        if self.lower_limit is not None:
            u_out = max(self.lower_limit, u_out)
        if self.upper_limit is not None:
            u_out = min(self.upper_limit, u_out)

        self.outputs[0] = u_out
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        e = self.inputs[0]
        x_i, x_d = self.states[0], self.states[1]

        # Anti-windup clamping on integrator state
        dx_i = e
        if self.upper_limit is not None and self.outputs[0] >= self.upper_limit and e > 0:
            dx_i = 0.0
        elif self.lower_limit is not None and self.outputs[0] <= self.lower_limit and e < 0:
            dx_i = 0.0

        dx_d = self.n_filter * (e - x_d)
        return np.array([dx_i, dx_d], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(2, dtype=float)


# =====================================================================
# 5. SIGNAL SOURCES & SINKS
# =====================================================================

class ConstantBlock(Block):
    """Constant Value Source."""

    def __init__(self, name: str, value: float = 1.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.value = float(value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.value
        return self.outputs


class StepSourceBlock(Block):
    """Step Input Source."""

    def __init__(self, name: str, step_time: float = 0.0, amplitude: float = 1.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.step_time = float(step_time)
        self.amplitude = float(amplitude)
        self.initial_value = float(initial_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.amplitude if t >= self.step_time else self.initial_value
        return self.outputs


class RampSourceBlock(Block):
    """Ramp Signal Source: y(t) = initial_value + slope * (t - start_time) for t >= start_time."""

    def __init__(self, name: str, slope: float = 1.0, start_time: float = 0.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.slope = float(slope)
        self.start_time = float(start_time)
        self.initial_value = float(initial_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        if t >= self.start_time:
            self.outputs[0] = self.initial_value + self.slope * (t - self.start_time)
        else:
            self.outputs[0] = self.initial_value
        return self.outputs


class SineSourceBlock(Block):
    """Sine Wave Generator."""

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


class PulseGeneratorBlock(Block):
    """Pulse Train Generator."""

    def __init__(self, name: str, period: float = 1.0, duty_cycle: float = 0.5, amplitude: float = 1.0, delay: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.period = max(1e-6, float(period))
        self.duty = max(0.0, min(1.0, float(duty_cycle)))
        self.amplitude = float(amplitude)
        self.delay = float(delay)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        if t < self.delay:
            self.outputs[0] = 0.0
            return self.outputs
        t_cycle = (t - self.delay) % self.period
        self.outputs[0] = self.amplitude if t_cycle < (self.period * self.duty) else 0.0
        return self.outputs


class BandLimitedWhiteNoiseBlock(Block):
    """Band-Limited Gaussian Random Noise Generator."""

    def __init__(self, name: str, noise_power: float = 0.1, sample_time: float = 0.01, seed: int = 42):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.noise_power = float(noise_power)
        self.sample_time = max(1e-6, float(sample_time))
        self.rng = np.random.RandomState(seed)
        self.curr_val = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            std = math.sqrt(self.noise_power / self.sample_time)
            self.curr_val = float(self.rng.normal(0.0, std))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.curr_val
        return self.outputs

    def reset_states(self) -> None:
        self.curr_val = 0.0
        self.last_sample_t = -1.0


class ScopeSinkBlock(Block):
    """Records signal history for high-resolution plotting and automated engineering metrics extraction."""

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
