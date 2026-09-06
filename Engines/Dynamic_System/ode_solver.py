"""Runge-Kutta 4th Order (RK4) continuous-time simulator for Dynamic Systems."""

from __future__ import annotations
from typing import Dict, Optional, Union
import numpy as np
from CORE.common_math import Waveform, parse_eng_unit
from .scheduler import SystemDiagram, Scheduler
from .blocks import ScopeSinkBlock


class DynamicSystemSimulator:
    """Solves block diagram ODE equations across time using RK4 integration."""

    def __init__(self, diagram: SystemDiagram):
        self.diagram = diagram
        self.scheduler = Scheduler(diagram)

    def simulate(
        self,
        t_stop: Union[str, float],
        dt: Union[str, float] = 1e-3,
        t_start: Union[str, float] = 0.0
    ) -> Dict[str, Waveform]:
        """Runs RK4 time integration loop and returns recorded Waveform traces."""
        t_end = parse_eng_unit(t_stop)
        step_size = parse_eng_unit(dt)
        t_curr = parse_eng_unit(t_start)

        self.scheduler.build_schedule()

        # Reset block states and scopes
        for b in self.diagram.blocks.values():
            b.reset_states()

        total_steps = int((t_end - t_curr) / step_size) + 1

        for step in range(total_steps):
            t = t_curr + step * step_size

            if len(self.scheduler.stateful_blocks) == 0:
                # Pure static / feedthrough system without states
                self.scheduler.propagate_signals(t)
                continue

            # RK4 Integration:
            # 1. k1 = f(t, x)
            x0 = self.scheduler.pack_states()
            k1 = self.scheduler.compute_all_derivatives(t)

            # 2. k2 = f(t + dt/2, x + dt/2 * k1)
            self.scheduler.unpack_states(x0 + 0.5 * step_size * k1)
            k2 = self.scheduler.compute_all_derivatives(t + 0.5 * step_size)

            # 3. k3 = f(t + dt/2, x + dt/2 * k2)
            self.scheduler.unpack_states(x0 + 0.5 * step_size * k2)
            k3 = self.scheduler.compute_all_derivatives(t + 0.5 * step_size)

            # 4. k4 = f(t + dt, x + dt * k3)
            self.scheduler.unpack_states(x0 + step_size * k3)
            k4 = self.scheduler.compute_all_derivatives(t + step_size)

            # Update state: x = x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            dx = (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            dx = np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)
            x_next = np.clip(x0 + dx, -1e9, 1e9)
            self.scheduler.unpack_states(x_next)

        # Collect Scope waveforms
        results: Dict[str, Waveform] = {}
        for b in self.diagram.blocks.values():
            if isinstance(b, ScopeSinkBlock):
                wf = Waveform(
                    b.time_history,
                    b.signal_history,
                    name=b.name,
                    x_unit="s",
                    y_unit="Output",
                    domain="time"
                )
                results[b.name] = wf

        return results
