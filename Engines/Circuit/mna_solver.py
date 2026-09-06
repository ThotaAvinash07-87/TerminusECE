"""Modified Nodal Analysis (MNA) solver for DC, AC, and Transient circuit simulations."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from CORE.common_math import parse_eng_unit, Waveform
from .components import (
    Component,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    VCVS,
    BJT,
)
from .netlist_parser import Netlist


class SimulationResult:
    """Holds time or frequency domain simulation output traces and reports."""

    def __init__(self, sim_type: str, x_vector: np.ndarray, x_label: str = "Time (s)"):
        self.sim_type = sim_type.upper()  # OP, DC, AC, TRAN
        self.x = np.asarray(x_vector, dtype=float)
        self.x_label = x_label
        # Map: trace_name -> Waveform
        self.waveforms: Dict[str, Waveform] = {}
        # Scalar operating point results if applicable
        self.op_results: Dict[str, float] = {}
        self.message: str = ""

    def add_waveform(self, name: str, y: np.ndarray, unit: str = "V", domain: str = "time") -> Waveform:
        x_unit = "Hz" if self.sim_type == "AC" else "s"
        wf = Waveform(self.x, y, name=name, x_unit=x_unit, y_unit=unit, domain=domain)
        self.waveforms[name] = wf
        return wf

    def get_waveform(self, name: str) -> Optional[Waveform]:
        return self.waveforms.get(name)

    def summary(self) -> str:
        """Returns brief text summary of simulation results."""
        lines = [f"=== Simulation Result: {self.sim_type} ==="]
        if self.message:
            lines.append(self.message)
        if self.sim_type == "OP":
            lines.append("Node Voltages & Branch Currents:")
            for k, v in sorted(self.op_results.items()):
                lines.append(f"  {k:15s} = {v:12.6g}")
        else:
            lines.append(f"Swept Points: {len(self.x)}")
            lines.append(f"Traces Recorded: {', '.join(self.waveforms.keys())}")
        return "\n".join(lines)


class MNASolver:
    """Solves circuits formulated as Modified Nodal Analysis linear/non-linear systems."""

    def __init__(self, netlist: Netlist):
        self.netlist = netlist

    def _build_node_index_map(self) -> Tuple[Dict[str, int], List[str]]:
        """Maps circuit nodes to 0-based matrix row indices (ground '0' is excluded from unknowns)."""
        all_nodes = self.netlist.get_all_nodes()
        node_to_idx: Dict[str, int] = {}
        idx_to_node: List[str] = []

        curr_idx = 0
        for n in all_nodes:
            norm = self.netlist.normalize_node(n)
            if norm == "0":
                continue
            if norm not in node_to_idx:
                node_to_idx[norm] = curr_idx
                idx_to_node.append(norm)
                curr_idx += 1

        return node_to_idx, idx_to_node

    def solve_op(self, max_iter: int = 50, tol: float = 1e-6) -> SimulationResult:
        """Calculates DC Operating Point (non-linear Newton-Raphson for active devices)."""
        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        # Collect voltage sources & inductors (shorts in DC) for auxiliary equations
        v_sources: List[VoltageSource] = []
        inductors: List[Inductor] = []
        diodes: List[Diode] = []

        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)
            elif isinstance(comp, Inductor):
                inductors.append(comp)
            elif isinstance(comp, Diode):
                diodes.append(comp)

        num_aux = len(v_sources) + len(inductors)
        total_dim = num_nodes + num_aux

        if total_dim == 0:
            res = SimulationResult("OP", np.array([0.0]))
            res.message = "Empty circuit netlist."
            return res

        # Assign aux indices
        for i, vs in enumerate(v_sources):
            vs.aux_index = num_nodes + i
        for i, ind in enumerate(inductors):
            ind.aux_index = num_nodes + len(v_sources) + i

        # Solution vector: [V_nodes (0..num_nodes-1), I_aux (num_nodes..total_dim-1)]
        sol = np.zeros(total_dim, dtype=float)

        # Newton-Raphson iteration loop
        converged = False
        for iteration in range(max_iter):
            A = np.zeros((total_dim, total_dim), dtype=float)
            b = np.zeros(total_dim, dtype=float)

            # 1. Stamp linear Resistors
            for comp in self.netlist.components.values():
                if isinstance(comp, Resistor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    g = comp.conductance
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)

                    if p_idx is not None:
                        A[p_idx, p_idx] += g
                    if n_idx is not None:
                        A[n_idx, n_idx] += g
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= g
                        A[n_idx, p_idx] -= g

            # 2. Stamp Independent Current Sources
            for comp in self.netlist.components.values():
                if isinstance(comp, CurrentSource):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    val = comp.dc

                    if p_idx is not None:
                        b[p_idx] -= val  # leaves positive node
                    if n_idx is not None:
                        b[n_idx] += val  # enters negative node

            # 3. Stamp Voltage Sources
            for vs in v_sources:
                np_name = self.netlist.normalize_node(vs.nodes[0])
                nn_name = self.netlist.normalize_node(vs.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                aux = vs.aux_index

                if p_idx is not None:
                    A[p_idx, aux] += 1.0
                    A[aux, p_idx] += 1.0
                if n_idx is not None:
                    A[n_idx, aux] -= 1.0
                    A[aux, n_idx] -= 1.0
                b[aux] += vs.dc

            # 4. Stamp Inductors (0V short across terminals in DC)
            for ind in inductors:
                np_name = self.netlist.normalize_node(ind.nodes[0])
                nn_name = self.netlist.normalize_node(ind.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                aux = ind.aux_index

                if p_idx is not None:
                    A[p_idx, aux] += 1.0
                    A[aux, p_idx] += 1.0
                if n_idx is not None:
                    A[n_idx, aux] -= 1.0
                    A[aux, n_idx] -= 1.0
                b[aux] += 0.0

            # 5. Stamp Non-linear Diodes
            for dio in diodes:
                np_name = self.netlist.normalize_node(dio.nodes[0])
                nn_name = self.netlist.normalize_node(dio.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)

                vp = sol[p_idx] if p_idx is not None else 0.0
                vn = sol[n_idx] if n_idx is not None else 0.0
                vd = vp - vn

                gd, ieq = dio.linearize(vd)

                if p_idx is not None:
                    A[p_idx, p_idx] += gd
                    b[p_idx] -= ieq
                if n_idx is not None:
                    A[n_idx, n_idx] += gd
                    b[n_idx] += ieq
                if p_idx is not None and n_idx is not None:
                    A[p_idx, n_idx] -= gd
                    A[n_idx, p_idx] -= gd

            # Add GMIN for floating node convergence stability
            for i in range(num_nodes):
                A[i, i] += 1e-12

            # Solve linear system
            try:
                new_sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                new_sol = np.linalg.lstsq(A, b, rcond=None)[0]

            diff = np.max(np.abs(new_sol - sol))
            sol = new_sol
            if diff < tol or len(diodes) == 0:
                converged = True
                break

        res = SimulationResult("OP", np.array([0.0]))
        res.message = f"DC Operating Point converged in {iteration + 1} iteration(s)."
        for name, idx in node_map.items():
            res.op_results[f"V({name})"] = float(sol[idx])
        for vs in v_sources:
            res.op_results[f"I({vs.name})"] = float(sol[vs.aux_index])
        for ind in inductors:
            res.op_results[f"I({ind.name})"] = float(sol[ind.aux_index])

        return res

    def solve_dc_sweep(
        self,
        src_name: str,
        start_val: float,
        stop_val: float,
        step_val: float
    ) -> SimulationResult:
        """Sweeps a DC source parameter and computes node voltages."""
        src_comp = self.netlist.components.get(src_name.upper())
        if not src_comp or not isinstance(src_comp, VoltageSource):
            raise KeyError(f"Voltage source '{src_name}' not found for DC sweep.")

        orig_val = src_comp.dc
        sweep_vals = np.arange(start_val, stop_val + step_val * 0.5, step_val)
        res = SimulationResult("DC", sweep_vals, x_label=f"{src_name} (V)")

        node_traces: Dict[str, List[float]] = {}

        for val in sweep_vals:
            src_comp.dc = float(val)
            op = self.solve_op()
            for k, v in op.op_results.items():
                if k not in node_traces:
                    node_traces[k] = []
                node_traces[k].append(v)

        src_comp.dc = orig_val
        for k, vlist in node_traces.items():
            res.add_waveform(k, np.array(vlist), unit="V" if k.startswith("V") else "A", domain="dc_sweep")

        return res

    def solve_ac(
        self,
        sweep_type: str = "dec",
        points: int = 10,
        f_start: Union[str, float] = 1.0,
        f_stop: Union[str, float] = 100e3
    ) -> SimulationResult:
        """AC Small-Signal Frequency Sweep Analysis."""
        f_start = parse_eng_unit(f_start)
        f_stop = parse_eng_unit(f_stop)

        if sweep_type.lower() == "dec":
            decades = math.log10(f_stop / f_start)
            num_pts = max(2, int(decades * points) + 1)
            frequencies = np.logspace(math.log10(f_start), math.log10(f_stop), num_pts)
        else:
            frequencies = np.linspace(f_start, f_stop, max(2, points))

        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        # Auxiliary components (Voltage sources, VCVS)
        v_sources: List[VoltageSource] = []
        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)

        num_aux = len(v_sources)
        total_dim = num_nodes + num_aux

        for i, vs in enumerate(v_sources):
            vs.aux_index = num_nodes + i

        node_complex_data: Dict[str, List[complex]] = {name: [] for name in node_map.keys()}

        for freq in frequencies:
            omega = 2.0 * math.pi * freq
            j_omega = 1j * omega

            A = np.zeros((total_dim, total_dim), dtype=complex)
            b = np.zeros(total_dim, dtype=complex)

            # Stamp Resistors
            for comp in self.netlist.components.values():
                if isinstance(comp, Resistor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    g = comp.conductance
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += g
                    if n_idx is not None:
                        A[n_idx, n_idx] += g
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= g
                        A[n_idx, p_idx] -= g

            # Stamp Capacitors: Y = j * omega * C
            for comp in self.netlist.components.values():
                if isinstance(comp, Capacitor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    yc = j_omega * comp.value
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += yc
                    if n_idx is not None:
                        A[n_idx, n_idx] += yc
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= yc
                        A[n_idx, p_idx] -= yc

            # Stamp Inductors: Y = 1 / (j * omega * L)
            for comp in self.netlist.components.values():
                if isinstance(comp, Inductor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    yl = 1.0 / (j_omega * comp.value + 1e-15)
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += yl
                    if n_idx is not None:
                        A[n_idx, n_idx] += yl
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= yl
                        A[n_idx, p_idx] -= yl

            # Stamp AC Voltage Sources
            for vs in v_sources:
                np_name = self.netlist.normalize_node(vs.nodes[0])
                nn_name = self.netlist.normalize_node(vs.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                aux = vs.aux_index

                if p_idx is not None:
                    A[p_idx, aux] += 1.0
                    A[aux, p_idx] += 1.0
                if n_idx is not None:
                    A[n_idx, aux] -= 1.0
                    A[aux, n_idx] -= 1.0

                # Phasor V = Mag * exp(j * Phase)
                phasor = vs.ac_mag * math.cos(math.radians(vs.ac_phase)) + 1j * vs.ac_mag * math.sin(math.radians(vs.ac_phase))
                b[aux] += phasor

            # GMIN diagonal stability
            for i in range(num_nodes):
                A[i, i] += 1e-12

            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]

            for name, idx in node_map.items():
                node_complex_data[name].append(sol[idx])

        result = SimulationResult("AC", frequencies, x_label="Frequency (Hz)")
        for name, cdata in node_complex_data.items():
            result.add_waveform(f"V({name})", np.array(cdata), unit="V", domain="frequency")

        return result

    def solve_tran(
        self,
        t_step: Union[str, float],
        t_stop: Union[str, float],
        t_start: Union[str, float] = 0.0,
        method: str = "trapezoidal"
    ) -> SimulationResult:
        """Transient Time-Domain Analysis using Companion Models."""
        dt = parse_eng_unit(t_step)
        t_end = parse_eng_unit(t_stop)
        t_0 = parse_eng_unit(t_start)

        time_points = np.arange(t_0, t_end + dt * 0.5, dt)
        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        v_sources: List[VoltageSource] = []
        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)

        num_aux = len(v_sources)
        total_dim = num_nodes + num_aux

        for i, vs in enumerate(v_sources):
            vs.aux_index = num_nodes + i

        # Initialize capacitors and inductors
        capacitors = [c for c in self.netlist.components.values() if isinstance(c, Capacitor)]
        inductors = [c for c in self.netlist.components.values() if isinstance(c, Inductor)]

        node_waveforms: Dict[str, List[float]] = {name: [] for name in node_map.keys()}

        # Initial conditions
        for cap in capacitors:
            cap.v_prev = cap.ic
            cap.i_prev = 0.0
        for ind in inductors:
            ind.i_prev = ind.ic
            ind.v_prev = 0.0

        for t in time_points:
            A = np.zeros((total_dim, total_dim), dtype=float)
            b = np.zeros(total_dim, dtype=float)

            # 1. Resistors
            for comp in self.netlist.components.values():
                if isinstance(comp, Resistor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    g = comp.conductance
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += g
                    if n_idx is not None:
                        A[n_idx, n_idx] += g
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= g
                        A[n_idx, p_idx] -= g

            # 2. Capacitors (Companion Model: G_eq, I_eq)
            for cap in capacitors:
                np_name = self.netlist.normalize_node(cap.nodes[0])
                nn_name = self.netlist.normalize_node(cap.nodes[1])
                geq, ieq = cap.get_companion_model(dt, method=method)
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                if p_idx is not None:
                    A[p_idx, p_idx] += geq
                    b[p_idx] += ieq
                if n_idx is not None:
                    A[n_idx, n_idx] += geq
                    b[n_idx] -= ieq
                if p_idx is not None and n_idx is not None:
                    A[p_idx, n_idx] -= geq
                    A[n_idx, p_idx] -= geq

            # 3. Inductors (Companion Model: G_eq, I_eq)
            for ind in inductors:
                np_name = self.netlist.normalize_node(ind.nodes[0])
                nn_name = self.netlist.normalize_node(ind.nodes[1])
                geq, ieq = ind.get_companion_model(dt, method=method)
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                if p_idx is not None:
                    A[p_idx, p_idx] += geq
                    b[p_idx] -= ieq
                if n_idx is not None:
                    A[n_idx, n_idx] += geq
                    b[n_idx] += ieq
                if p_idx is not None and n_idx is not None:
                    A[p_idx, n_idx] -= geq
                    A[n_idx, p_idx] -= geq

            # 4. Voltage Sources (time evaluated)
            for vs in v_sources:
                np_name = self.netlist.normalize_node(vs.nodes[0])
                nn_name = self.netlist.normalize_node(vs.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                aux = vs.aux_index
                if p_idx is not None:
                    A[p_idx, aux] += 1.0
                    A[aux, p_idx] += 1.0
                if n_idx is not None:
                    A[n_idx, aux] -= 1.0
                    A[aux, n_idx] -= 1.0
                b[aux] += vs.value_at_time(t)

            # GMIN diagonal stability
            for i in range(num_nodes):
                A[i, i] += 1e-12

            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]

            # Record node voltages
            for name, idx in node_map.items():
                node_waveforms[name].append(float(sol[idx]))

            # Update capacitor states for next step
            for cap in capacitors:
                np_name = self.netlist.normalize_node(cap.nodes[0])
                nn_name = self.netlist.normalize_node(cap.nodes[1])
                vp = sol[node_map[np_name]] if np_name in node_map else 0.0
                vn = sol[node_map[nn_name]] if nn_name in node_map else 0.0
                v_cap = vp - vn
                geq, ieq = cap.get_companion_model(dt, method=method)
                cap.i_prev = geq * v_cap - ieq
                cap.v_prev = v_cap

            # Update inductor states
            for ind in inductors:
                np_name = self.netlist.normalize_node(ind.nodes[0])
                nn_name = self.netlist.normalize_node(ind.nodes[1])
                vp = sol[node_map[np_name]] if np_name in node_map else 0.0
                vn = sol[node_map[nn_name]] if nn_name in node_map else 0.0
                v_ind = vp - vn
                geq, ieq = ind.get_companion_model(dt, method=method)
                ind.i_prev = geq * v_ind + ieq
                ind.v_prev = v_ind

        result = SimulationResult("TRAN", time_points, x_label="Time (s)")
        for name, vdata in node_waveforms.items():
            result.add_waveform(f"V({name})", np.array(vdata), unit="V", domain="time")

        return result
