"""Circuit component definitions for MNA (Modified Nodal Analysis) solver."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union
from CORE.common_math import parse_eng_unit


class Component:
    """Base class for all circuit components."""
    
    def __init__(self, name: str, nodes: List[str]):
        self.name = name.strip()
        self.nodes = [n.strip() for n in nodes]
        self.params: Dict[str, float] = {}

    def get_node(self, terminal_index: int) -> str:
        if 0 <= terminal_index < len(self.nodes):
            return self.nodes[terminal_index]
        return "0"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} nodes={self.nodes}>"


class Resistor(Component):
    """Linear Resistor (R)."""
    
    def __init__(self, name: str, node_p: str, node_n: str, resistance: Union[str, float]):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(resistance)
        if self.value <= 0:
            raise ValueError(f"Resistance for {name} must be positive, got {self.value}")

    @property
    def conductance(self) -> float:
        return 1.0 / self.value


class Capacitor(Component):
    """Linear Capacitor (C) with companion model integration states."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        capacitance: Union[str, float],
        ic: float = 0.0
    ):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(capacitance)
        self.ic = ic  # Initial condition voltage
        self.v_prev = ic
        self.i_prev = 0.0

    def get_companion_model(self, dt: float, method: str = "trapezoidal") -> Tuple[float, float]:
        """Calculates equivalent conductance G_eq and current source I_eq for time-stepping."""
        if method == "backward_euler":
            geq = self.value / dt
            ieq = geq * self.v_prev
        else:  # trapezoidal
            geq = 2.0 * self.value / dt
            ieq = geq * self.v_prev + self.i_prev
        return geq, ieq


class Inductor(Component):
    """Linear Inductor (L) with companion model integration states."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        inductance: Union[str, float],
        ic: float = 0.0
    ):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(inductance)
        self.ic = ic  # Initial current
        self.v_prev = 0.0
        self.i_prev = ic
        self.aux_index: int = -1  # Assigned by MNA solver

    def get_companion_model(self, dt: float, method: str = "trapezoidal") -> Tuple[float, float]:
        """Calculates equivalent conductance G_eq and current source I_eq."""
        if method == "backward_euler":
            geq = dt / self.value
            ieq = self.i_prev
        else:  # trapezoidal
            geq = dt / (2.0 * self.value)
            ieq = self.i_prev + geq * self.v_prev
        return geq, ieq


class VoltageSource(Component):
    """Independent Voltage Source (DC, AC, Sine, Pulse)."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        dc: Union[str, float] = 0.0,
        ac_mag: float = 0.0,
        ac_phase: float = 0.0,
        waveform_type: str = "DC",
        wave_params: Optional[Dict[str, float]] = None
    ):
        super().__init__(name, [node_p, node_n])
        self.dc = parse_eng_unit(dc)
        self.ac_mag = parse_eng_unit(ac_mag) if ac_mag else 0.0
        self.ac_phase = float(ac_phase)
        self.waveform_type = waveform_type.upper()
        self.wave_params = wave_params or {}
        self.aux_index: int = -1  # Row/col in MNA matrix

    def value_at_time(self, t: float) -> float:
        """Evaluates time-dependent voltage."""
        if self.waveform_type == "DC":
            return self.dc
        
        elif self.waveform_type in ("SINE", "SIN"):
            offset = self.wave_params.get("offset", self.dc)
            amplitude = self.wave_params.get("amplitude", 1.0)
            freq = self.wave_params.get("freq", 1000.0)
            phase = self.wave_params.get("phase", 0.0)
            return offset + amplitude * math.sin(2.0 * math.pi * freq * t + math.radians(phase))
        
        elif self.waveform_type == "PULSE":
            v1 = self.wave_params.get("v1", 0.0)
            v2 = self.wave_params.get("v2", self.dc if self.dc != 0 else 5.0)
            td = self.wave_params.get("td", 0.0)
            tr = max(1e-15, self.wave_params.get("tr", 1e-9))
            tf = max(1e-15, self.wave_params.get("tf", 1e-9))
            ton = self.wave_params.get("ton", 1e-3)
            period = self.wave_params.get("period", 2e-3)

            if t < td:
                return v1
            t_rel = (t - td) % period
            if t_rel < tr:
                return v1 + (v2 - v1) * (t_rel / tr)
            elif t_rel < tr + ton:
                return v2
            elif t_rel < tr + ton + tf:
                return v2 - (v2 - v1) * ((t_rel - (tr + ton)) / tf)
            else:
                return v1

        return self.dc


class CurrentSource(Component):
    """Independent Current Source (enters node_p, leaves node_n)."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        dc: Union[str, float] = 0.0,
        ac_mag: float = 0.0,
        ac_phase: float = 0.0
    ):
        super().__init__(name, [node_p, node_n])
        self.dc = parse_eng_unit(dc)
        self.ac_mag = parse_eng_unit(ac_mag) if ac_mag else 0.0
        self.ac_phase = float(ac_phase)

    def value_at_time(self, t: float) -> float:
        return self.dc


class Diode(Component):
    """Semiconductor Diode using Shockley equation: Id = Is * (exp(Vd / (n*Vt)) - 1)."""
    
    def __init__(
        self,
        name: str,
        node_anode: str,
        node_cathode: str,
        is_sat: float = 1e-14,
        n_ideal: float = 1.0,
        vt: float = 0.02585,  # ~26mV at 300K
        rs: float = 0.001
    ):
        super().__init__(name, [node_anode, node_cathode])
        self.is_sat = is_sat
        self.n_ideal = n_ideal
        self.vt = vt
        self.rs = rs
        self.vd_prev = 0.6  # Initial guess

    def linearize(self, vd: float) -> Tuple[float, float]:
        """Calculates linearized companion conductance gd and current id_eq for Newton-Raphson iteration.
        
        i = Is*(exp(vd/(n*Vt)) - 1)
        gd = d(i)/d(vd) = (Is / (n*Vt)) * exp(vd / (n*Vt))
        ieq = i - gd * vd
        """
        vd_limited = max(-10.0, min(vd, 1.2))  # Voltage limiting for numerical stability
        nvt = self.n_ideal * self.vt
        exp_term = math.exp(vd_limited / nvt)
        id_val = self.is_sat * (exp_term - 1.0)
        gd = max(1e-12, (self.is_sat / nvt) * exp_term)
        ieq = id_val - gd * vd_limited
        return gd, ieq


class VCVS(Component):
    """Voltage-Controlled Voltage Source (E-source or Op-Amp)."""
    
    def __init__(
        self,
        name: str,
        node_out_p: str,
        node_out_n: str,
        node_in_p: str,
        node_in_n: str,
        gain: Union[str, float] = 1.0
    ):
        super().__init__(name, [node_out_p, node_out_n, node_in_p, node_in_n])
        self.gain = parse_eng_unit(gain)
        self.aux_index: int = -1


class BJT(Component):
    """Bipolar Junction Transistor (NPN / PNP simplified model)."""
    
    def __init__(
        self,
        name: str,
        node_c: str,
        node_b: str,
        node_e: str,
        bjt_type: str = "NPN",
        beta: float = 100.0,
        vbe_on: float = 0.7
    ):
        super().__init__(name, [node_c, node_b, node_e])
        self.bjt_type = bjt_type.upper()
        self.beta = beta
        self.vbe_on = vbe_on
