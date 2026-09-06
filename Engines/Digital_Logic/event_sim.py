"""Discrete Event Simulation engine and ASCII timing waveform generator for digital logic."""

from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Set, Tuple, Union
from .gates import LogicValue, LogicGate, DFlipFlop, ClockGenerator
from .hdl_parser import LogicCircuit


class EventSimulator:
    """Priority-Queue driven discrete event simulator for digital circuits."""

    def __init__(self, circuit: LogicCircuit):
        self.circuit = circuit
        self.wire_states: Dict[str, LogicValue] = {w: LogicValue.LOW for w in circuit.wires}
        # History: wire -> list of (time_ns, value)
        self.traces: Dict[str, List[Tuple[float, LogicValue]]] = {w: [(0.0, LogicValue.LOW)] for w in circuit.wires}
        # Priority queue: (time_ns, sequence_id, wire_name, new_val)
        self.event_queue: List[Tuple[float, int, str, LogicValue]] = []
        self._seq = 0

    def schedule_event(self, time_ns: float, wire_name: str, val: LogicValue) -> None:
        self._seq += 1
        heapq.heappush(self.event_queue, (time_ns, self._seq, wire_name, val))

    def run(self, max_time_ns: float = 100.0) -> Dict[str, List[Tuple[float, LogicValue]]]:
        """Runs the event-driven simulation loop until max_time_ns."""
        # 1. Schedule initial clock toggle events
        for cname, clk in self.circuit.clocks.items():
            t = 0.0
            val = LogicValue.LOW
            while t <= max_time_ns:
                self.schedule_event(t, cname, val)
                duration = clk.high_time if val == LogicValue.HIGH else clk.low_time
                t += duration
                val = LogicValue.HIGH if val == LogicValue.LOW else LogicValue.LOW

        # 2. Main event loop
        while self.event_queue:
            t, _, wire, new_val = heapq.heappop(self.event_queue)
            if t > max_time_ns:
                break

            current_val = self.wire_states.get(wire, LogicValue.LOW)
            if new_val != current_val:
                self.wire_states[wire] = new_val
                self.traces[wire].append((t, new_val))

                # Trigger downstream gates
                for gname, gate in self.circuit.gates.items():
                    if wire in gate.input_wires:
                        # Evaluate gate
                        in_vals = [self.wire_states.get(w, LogicValue.LOW) for w in gate.input_wires]
                        out_val = gate.evaluate(in_vals)
                        out_wire = gate.output_wire
                        target_t = t + gate.delay_ns
                        self.schedule_event(target_t, out_wire, out_val)

                # Trigger downstream flip-flops
                for ff_name, ff in self.circuit.flip_flops.items():
                    if isinstance(ff, DFlipFlop):
                        if wire == ff.clk_wire or wire == ff.rst_wire:
                            d_val = self.wire_states.get(ff.d_wire, LogicValue.LOW)
                            clk_val = self.wire_states.get(ff.clk_wire, LogicValue.LOW)
                            rst_val = self.wire_states.get(ff.rst_wire) if ff.rst_wire else None
                            new_q, new_q_bar = ff.clock_transition(d_val, clk_val, rst_val)

                            target_t = t + ff.delay_ns
                            self.schedule_event(target_t, ff.q_wire, new_q)
                            if ff.q_bar_wire and new_q_bar is not None:
                                self.schedule_event(target_t, ff.q_bar_wire, new_q_bar)

        return self.traces


class DigitalWaveformTracer:
    """Renders multi-channel digital logic timing diagrams in ASCII / Unicode."""

    @classmethod
    def render_timing_diagram(
        cls,
        traces: Dict[str, List[Tuple[float, LogicValue]]],
        max_time_ns: float = 100.0,
        time_step_ns: float = 2.0,
        active_wires: Optional[List[str]] = None
    ) -> str:
        """Renders timing trace ASCII lines: `_┌──┐_┌──┐_`."""
        wires = active_wires or list(traces.keys())
        num_cols = int(max_time_ns / time_step_ns) + 1

        lines: List[str] = []
        header = f"  {'Time (ns)':12s}: " + "".join(f"{int(c * time_step_ns):<4}" for c in range(0, num_cols, 4))
        lines.append(header)
        lines.append("  " + "─" * (len(header) + 5))

        for wire in wires:
            trace = traces.get(wire, [(0.0, LogicValue.LOW)])
            char_row = []
            prev_val = LogicValue.LOW

            for col in range(num_cols):
                t_col = col * time_step_ns
                # Find latest value at or before t_col
                current_val = LogicValue.LOW
                for t_ev, v in trace:
                    if t_ev <= t_col:
                        current_val = v
                    else:
                        break

                if current_val == LogicValue.HIGH:
                    if prev_val == LogicValue.LOW:
                        char_row.append("┌")
                    else:
                        char_row.append("─")
                elif current_val == LogicValue.LOW:
                    if prev_val == LogicValue.HIGH:
                        char_row.append("┘")
                    else:
                        char_row.append("_")
                else:
                    char_row.append("X")

                prev_val = current_val

            row_str = "".join(char_row)
            lines.append(f"  {wire:12s}: {row_str}")

        return "\n".join(lines)
