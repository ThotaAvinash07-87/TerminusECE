"""Main Textual application and unified command router for TerminusECE."""

from __future__ import annotations
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from CORE.common_math import parse_eng_unit, format_eng_unit, Waveform, SignalMetrics, split_smart_statements
from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter, SchematicVisualizer
from CORE.ipc_router import IPCRouter, IPCClient

from Engines.Circuit.components import (
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Diode, VCVS, BJT
)
from Engines.Circuit.netlist_parser import Netlist, CircuitParser
from Engines.Circuit.mna_solver import MNASolver, SimulationResult

from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from Engines.Numerical.transforms import TransferFunction, DiscreteTransferFunction

from Engines.Dynamic_System.blocks import (
    Block, IntegratorBlock, DerivativeBlock, TransferFunctionBlock, StateSpaceBlock,
    TransportDelayBlock, SaturationBlock, RateLimiterBlock, DeadZoneBlock, BacklashBlock,
    RelayBlock, CoulombViscousFrictionBlock, QuantizerBlock, GainBlock, SumBlock,
    ProductBlock, MathFunctionBlock, LookupTable1DBlock, SwitchBlock,
    ZeroOrderHoldBlock, UnitDelayBlock, PIDBlock, ConstantBlock,
    StepSourceBlock, RampSourceBlock, SineSourceBlock, PulseGeneratorBlock,
    BandLimitedWhiteNoiseBlock, ScopeSinkBlock
)
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator

from Engines.Digital_Logic.gates import LogicValue
from Engines.Digital_Logic.hdl_parser import LogicCircuit, HDLParser
from Engines.Digital_Logic.event_sim import EventSimulator, DigitalWaveformTracer

from Engines.Embedded.mcu_core import MCUCore
from Engines.Embedded.toolchain import Assembler, Disassembler


class TerminusEngineBridge:
    """Decoupled computational backend bridge that holds all engine states and executes commands."""

    def __init__(self):
        # Current active mode: 'CIRCUIT', 'NUMERICAL', 'DYNAMIC', 'DIGITAL', 'EMBEDDED', 'UNIFIED'
        self.mode = "CIRCUIT"

        # Engines
        self.circuit_netlist = Netlist()
        self.circuit_solver = MNASolver(self.circuit_netlist)
        self.last_circuit_sim: Optional[SimulationResult] = None

        self.numerical_workspace = NumericalWorkspace()
        self.numerical_parser = NumericalASTParser(self.numerical_workspace)

        self.dynamic_diagram = SystemDiagram()
        self.dynamic_simulator = DynamicSystemSimulator(self.dynamic_diagram)
        self.last_dynamic_sim: Optional[Dict[str, Waveform]] = None

        self.logic_circuit = LogicCircuit()
        self.last_logic_traces: Optional[Dict[str, List[Tuple[float, LogicValue]]]] = None

        self.mcu = MCUCore()

        # IPC
        self.ipc_client = IPCClient()
        self.ipc_router: Optional[IPCRouter] = None

        # Global Signal / Variable bus for piping
        self.global_store: Dict[str, Any] = {}

    def switch_mode(self, mode_str: str) -> str:
        m = mode_str.lower().strip()
        if m in ("circuit", "ltspice", "spice"):
            self.mode = "CIRCUIT"
        elif m in ("numerical", "matlab", "matrix", "math"):
            self.mode = "NUMERICAL"
        elif m in ("dynamic", "simulink", "control", "systems"):
            self.mode = "DYNAMIC"
        elif m in ("digital", "xilinx", "logic", "verilog", "hdl"):
            self.mode = "DIGITAL"
        elif m in ("embedded", "mcu", "c2000", "dsp"):
            self.mode = "EMBEDDED"
        elif m in ("unified", "workbench", "all"):
            self.mode = "UNIFIED"
        else:
            raise ValueError(f"Unknown mode '{mode_str}'. Valid modes: circuit, numerical, dynamic, digital, embedded, unified")
        return self.mode

    def execute_command(self, raw_command: str) -> str:
        """Executes a text command line and returns formatted output string."""
        line = raw_command.strip()
        if not line or line.startswith("#"):
            return ""

        # Semicolon-separated batch commands
        sub_cmds = split_smart_statements(line, ";")
        if len(sub_cmds) > 1:
            outputs = []
            for sc in sub_cmds:
                out = self.execute_command(sc)
                if out:
                    outputs.append(out)
            return "\n".join(outputs)

        # Pipe commands
        if "|" in line and not line.lower().startswith("connect") and not line.lower().startswith("truth"):
            # Check if this is a pipeline: cmd1 | cmd2
            pipe_parts = [p.strip() for p in line.split("|")]
            # If not a connect command, execute sequentially
            last_out = ""
            for p in pipe_parts:
                last_out = self.execute_command(p)
            return last_out

        tokens = line.split()
        first = tokens[0].lower()

        # Global commands
        if first == "help":
            return self._cmd_help()

        if first == "mode":
            if len(tokens) < 2:
                return f"Current mode: {self.mode}. Options: circuit, numerical, dynamic, digital, embedded, unified"
            new_mode = self.switch_mode(tokens[1])
            return f"Context switched to: [bold magenta]{new_mode}[/bold magenta]"

        if first == "export":
            return self._cmd_export(tokens[1:])

        if first == "ipc":
            return self._cmd_ipc(tokens[1:])

        # Route to mode-specific handler
        if self.mode == "CIRCUIT":
            return self._handle_circuit(line, tokens)
        elif self.mode == "NUMERICAL":
            return self._handle_numerical(line, tokens)
        elif self.mode == "DYNAMIC":
            return self._handle_dynamic(line, tokens)
        elif self.mode == "DIGITAL":
            return self._handle_digital(line, tokens)
        elif self.mode == "EMBEDDED":
            return self._handle_embedded(line, tokens)
        else:
            return self._handle_unified(line, tokens)

    def _cmd_help(self) -> str:
        help_texts = [
            f"[bold cyan]=== TerminusECE Commands ({self.mode} Mode) ===[/bold cyan]",
            "Global Commands:",
            "  mode <subsystem>            - Switch mode (circuit, numerical, dynamic, digital, embedded)",
            "  export report [--format=csv]- Export last simulation results",
            "  ipc start / set / get       - Cross-terminal IPC sync",
        ]
        if self.mode == "CIRCUIT":
            help_texts.extend([
                "\nCircuit Commands (LTspice-like):",
                "  add <Name> <Val> [ac=1]      - e.g. add V1 10V ac=1, add R1 1k, add C1 1u",
                "  connect <p1> | <p2> | <net>  - e.g. connect V1.p | R1.a, connect R1.b | C1.a | node_out",
                "  run .ac dec 10 1Hz 100kHz    - Run AC frequency sweep & render ASCII Bode plot",
                "  run .tran 1u 10m             - Run Transient simulation",
                "  run .op                      - Run DC Operating Point",
                "  run .dc V1 0 10 0.1          - Run DC Sweep",
                "  list / clear                 - Show or reset netlist",
            ])
        elif self.mode == "NUMERICAL":
            help_texts.extend([
                "\nNumerical Commands (MATLAB-like):",
                "  A = [1 2; 3 4]               - Matrix definition",
                "  inv(A), det(A), eig(A)       - Linear algebra",
                "  H = tf([1], [1, 2, 1])       - Continuous Transfer Function H(s)",
                "  bode(H)                      - ASCII Bode plot",
                "  step(H)                      - Step response",
                "  whos / clear                 - Variable introspection",
            ])
        elif self.mode == "DYNAMIC":
            help_texts.extend([
                "\nDynamic Systems Commands (Simulink-like):",
                "  add <type> <name> [params]   - e.g. add step Step1, add gain G1 5, add tf Plant [1] [1 2 1], add scope Scope1",
                "  connect <B1.out> <B2.in>     - Route signal wire between blocks",
                "  sim <stop_time> [dt]         - Run RK4 ODE simulation & plot scopes",
                "  clear                        - Clear diagram",
            ])
        elif self.mode == "DIGITAL":
            help_texts.extend([
                "\nDigital Logic Commands (Xilinx-like):",
                "  wire a, b, c, clk, q         - Declare wires",
                "  gate G1 AND a b -> out       - Add logic gate",
                "  dff D1 clk=clk d=a q=q       - Add D-Flip-Flop",
                "  clock CLK period=10ns        - Add clock source",
                "  sim <max_time_ns>            - Run discrete event simulation & timing diagram",
                "  truth <boolean_expression>   - Generate truth table",
            ])
        elif self.mode == "EMBEDDED":
            help_texts.extend([
                "\nEmbedded Commands (C2000 MCU):",
                "  load <asm_code_or_file>      - Assemble and load program",
                "  step                         - Single-step instruction cycle",
                "  run [max_cycles]             - Run execution",
                "  dump                         - Dump registers, flags, and memory",
                "  pwm <period> <duty>          - Configure ePWM peripheral",
            ])
        return "\n".join(help_texts)

    def _cmd_export(self, args: List[str]) -> str:
        fmt = "csv"
        for a in args:
            if a.startswith("--format="):
                fmt = a.split("=")[1].lower()

        if self.last_circuit_sim:
            lines = [f"# TerminusECE Export - {self.last_circuit_sim.sim_type}"]
            for name, wf in self.last_circuit_sim.waveforms.items():
                lines.append(f"\n--- {name} ---")
                lines.append(wf.to_csv())
            return "\n".join(lines)
        return "No simulation data available to export."

    def _cmd_ipc(self, args: List[str]) -> str:
        if not args:
            return "Usage: ipc <start|set|get|list> [args]"
        sub = args[0].lower()
        if sub == "start":
            if self.ipc_router is None:
                self.ipc_router = IPCRouter()
                self.ipc_router.start_background()
                return "IPC Daemon started on 127.0.0.1:8765"
            return "IPC Daemon is already running."
        elif sub == "set" and len(args) >= 3:
            name = args[1]
            val = args[2]
            ok = self.ipc_client.set_variable(name, val)
            return f"IPC Set '{name}' = {val} ({'OK' if ok else 'Failed'})"
        elif sub == "get" and len(args) >= 2:
            name = args[1]
            val = self.ipc_client.get_variable(name)
            return f"IPC Get '{name}' -> {val}"
        return "Unknown IPC sub-command."

    # --- Circuit Handlers ---
    def _handle_circuit(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        if first in ("add", "connect", "remove", "delete", "set", "list", "show", "clear"):
            res = CircuitParser.parse_command(self.circuit_netlist, line)
            t = res.get("type")
            if t == "add":
                return f"[green]Added component:[/green] {res.get('name')}"
            elif t == "connect":
                return f"[green]Connected:[/green] {' | '.join(res.get('endpoints', []))}"
            elif t == "list":
                comps = res.get("components", [])
                if not comps:
                    return "Netlist is empty."
                return "[bold cyan]Active Netlist Components:[/bold cyan]\n" + "\n".join(f"  {c}" for c in comps)
            elif t == "clear":
                return "Circuit netlist cleared."
            elif t == "set":
                return f"Updated {res.get('target')} = {res.get('value')}"
            return str(res)

        elif first == "run":
            sim_spec = line[4:].strip()
            return self._run_circuit_simulation(sim_spec)

        # Fallback to direct run if starts with '.'
        if line.startswith("."):
            return self._run_circuit_simulation(line)

        raise ValueError(f"Unknown circuit command '{line}'. Type 'help' for options.")

    def _run_circuit_simulation(self, spec: str) -> str:
        parts = spec.split()
        if not parts:
            raise ValueError("No simulation specified. Example: 'run .ac dec 10 1Hz 100kHz'")

        # Generate ASCII schematic topology visualization of the circuit
        topo_diagram = SchematicVisualizer.render_circuit_topology(
            self.circuit_netlist.components,
            self.circuit_netlist.pin_map
        )

        sim_cmd = parts[0].lower()
        if sim_cmd == ".op":
            res = self.circuit_solver.solve_op()
            self.last_circuit_sim = res
            return f"{topo_diagram}\n\n{res.summary()}"

        elif sim_cmd == ".dc":
            if len(parts) < 5:
                raise ValueError("Usage: run .dc <Source> <Start> <Stop> <Step>")
            src = parts[1]
            start = parse_eng_unit(parts[2])
            stop = parse_eng_unit(parts[3])
            step = parse_eng_unit(parts[4])
            res = self.circuit_solver.solve_dc_sweep(src, start, stop, step)
            self.last_circuit_sim = res
            # Render plot of first non-source trace
            out_traces = [wf for k, wf in res.waveforms.items() if not k.startswith("I(")]
            plot_str = ""
            if out_traces:
                plot_str = AsciiPlotter.plot(out_traces[0].x, out_traces[0].y, title=f"DC Sweep: {out_traces[0].name}", x_label=f"{src} (V)")
            return f"{topo_diagram}\n\n{res.summary()}\n\n{plot_str}"

        elif sim_cmd == ".ac":
            # .ac dec 10 1Hz 100kHz
            sweep_type = "dec"
            pts = 10
            f_start = 1.0
            f_stop = 100e3

            if len(parts) >= 5:
                sweep_type = parts[1]
                pts = int(parts[2])
                f_start = parse_eng_unit(parts[3])
                f_stop = parse_eng_unit(parts[4])

            res = self.circuit_solver.solve_ac(sweep_type, pts, f_start, f_stop)
            self.last_circuit_sim = res

            # Render Bode plot for the last computed output node
            out_wf = None
            for name, wf in res.waveforms.items():
                if name != "V(0)":
                    out_wf = wf

            bode_str = ""
            if out_wf:
                bode_str = AsciiBodePlotter.plot_bode(
                    out_wf.x,
                    out_wf.magnitude_db,
                    out_wf.phase_deg,
                    title=f"Bode Plot: {out_wf.name}"
                )

            return f"{topo_diagram}\n\n{res.summary()}\n\n{bode_str}"

        elif sim_cmd == ".tran":
            # .tran 1u 10m [0]
            if len(parts) < 3:
                raise ValueError("Usage: run .tran <t_step> <t_stop> [t_start]")
            dt = parse_eng_unit(parts[1])
            t_stop = parse_eng_unit(parts[2])
            t_start = parse_eng_unit(parts[3]) if len(parts) > 3 else 0.0

            res = self.circuit_solver.solve_tran(dt, t_stop, t_start)
            self.last_circuit_sim = res

            out_wf = None
            for name, wf in res.waveforms.items():
                if name != "V(0)":
                    out_wf = wf

            plot_str = ""
            if out_wf:
                plot_str = AsciiPlotter.plot(out_wf.x, out_wf.y, title=f"Transient Response: {out_wf.name}", x_label="Time (s)", y_label="Voltage (V)")

            return f"{topo_diagram}\n\n{res.summary()}\n\n{plot_str}"

        raise ValueError(f"Unknown simulation command '{sim_cmd}'")

    # --- Numerical Handlers ---
    def _handle_numerical(self, line: str, tokens: List[str]) -> str:
        # Special plot functions: bode(H), step(H)
        if line.startswith("bode(") and line.endswith(")"):
            var_name = line[5:-1].strip()
            obj = self.numerical_workspace.variables.get(var_name)
            if isinstance(obj, TransferFunction):
                return obj.render_bode_ascii()
            raise ValueError(f"Variable '{var_name}' is not a TransferFunction.")

        if line.startswith("step(") and line.endswith(")"):
            var_name = line[5:-1].strip()
            obj = self.numerical_workspace.variables.get(var_name)
            if isinstance(obj, TransferFunction):
                wf = obj.step_response()
                return AsciiPlotter.plot(wf.x, wf.y, title=f"Step Response: {var_name}", x_label="Time (s)", y_label="Amplitude")
            raise ValueError(f"Variable '{var_name}' is not a TransferFunction.")

        val = self.numerical_parser.execute(line)
        if val is None:
            return ""
        if isinstance(val, (np.ndarray, list)):
            arr = np.asarray(val)
            if arr.ndim <= 2:
                return str(arr)
        return str(val)

    # --- Dynamic System Handlers ---
    def _handle_dynamic(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()
        if first == "clear":
            self.dynamic_diagram.clear()
            return "Dynamic system diagram cleared."

        if first == "add":
            # add <type> <name> [params]
            if len(tokens) < 3:
                raise ValueError("Usage: add <type> <name> [params...]")
            btype = tokens[1].lower()
            bname = tokens[2].upper()

            if btype in ("integrator", "int", "1/s"):
                ic = float(tokens[3]) if len(tokens) > 3 else 0.0
                lower = float(tokens[4]) if len(tokens) > 4 and tokens[4] != "none" else None
                upper = float(tokens[5]) if len(tokens) > 5 and tokens[5] != "none" else None
                self.dynamic_diagram.add_block(IntegratorBlock(bname, initial_condition=ic, lower_limit=lower, upper_limit=upper))
            elif btype in ("derivative", "deriv", "s"):
                tau = float(tokens[3]) if len(tokens) > 3 else 0.01
                self.dynamic_diagram.add_block(DerivativeBlock(bname, tau=tau))
            elif btype == "gain":
                gain = parse_eng_unit(tokens[3]) if len(tokens) > 3 else 1.0
                self.dynamic_diagram.add_block(GainBlock(bname, gain=gain))
            elif btype == "sum":
                signs = tokens[3] if len(tokens) > 3 else "+-"
                self.dynamic_diagram.add_block(SumBlock(bname, signs=signs))
            elif btype in ("product", "prod", "mult", "div"):
                ops = tokens[3] if len(tokens) > 3 else "**"
                self.dynamic_diagram.add_block(ProductBlock(bname, operations=ops))
            elif btype in ("mathfunc", "math", "func"):
                func = tokens[3] if len(tokens) > 3 else "sin"
                self.dynamic_diagram.add_block(MathFunctionBlock(bname, function=func))
            elif btype in ("saturation", "sat", "clamp"):
                lower = float(tokens[3]) if len(tokens) > 3 else -1.0
                upper = float(tokens[4]) if len(tokens) > 4 else 1.0
                self.dynamic_diagram.add_block(SaturationBlock(bname, lower_limit=lower, upper_limit=upper))
            elif btype in ("ratelimiter", "ratelimit", "slew"):
                rising = float(tokens[3]) if len(tokens) > 3 else 100.0
                falling = float(tokens[4]) if len(tokens) > 4 else -100.0
                self.dynamic_diagram.add_block(RateLimiterBlock(bname, rising_slew_rate=rising, falling_slew_rate=falling))
            elif btype in ("deadzone", "deadband"):
                start_z = float(tokens[3]) if len(tokens) > 3 else -0.5
                end_z = float(tokens[4]) if len(tokens) > 4 else 0.5
                self.dynamic_diagram.add_block(DeadZoneBlock(bname, start_zone=start_z, end_zone=end_z))
            elif btype in ("backlash", "hysteresis"):
                db = float(tokens[3]) if len(tokens) > 3 else 1.0
                self.dynamic_diagram.add_block(BacklashBlock(bname, deadband_width=db))
            elif btype in ("relay", "schmitt", "bangbang"):
                on_th = float(tokens[3]) if len(tokens) > 3 else 0.5
                off_th = float(tokens[4]) if len(tokens) > 4 else -0.5
                y_on = float(tokens[5]) if len(tokens) > 5 else 1.0
                y_off = float(tokens[6]) if len(tokens) > 6 else 0.0
                self.dynamic_diagram.add_block(RelayBlock(bname, switch_on_point=on_th, switch_off_point=off_th, output_on=y_on, output_off=y_off))
            elif btype in ("friction", "fric"):
                fc = float(tokens[3]) if len(tokens) > 3 else 1.0
                bv = float(tokens[4]) if len(tokens) > 4 else 0.1
                fs = float(tokens[5]) if len(tokens) > 5 else 1.5
                self.dynamic_diagram.add_block(CoulombViscousFrictionBlock(bname, f_coulomb=fc, b_viscous=bv, f_static=fs))
            elif btype in ("quantizer", "quant", "adc_dac"):
                q = float(tokens[3]) if len(tokens) > 3 else 0.1
                self.dynamic_diagram.add_block(QuantizerBlock(bname, quantization_interval=q))
            elif btype in ("delay", "transportdelay", "timedelay"):
                dt_delay = float(tokens[3]) if len(tokens) > 3 else 0.1
                self.dynamic_diagram.add_block(TransportDelayBlock(bname, delay_time=dt_delay))
            elif btype in ("zoh", "sampleandhold"):
                ts = float(tokens[3]) if len(tokens) > 3 else 0.01
                self.dynamic_diagram.add_block(ZeroOrderHoldBlock(bname, sample_time=ts))
            elif btype in ("unitdelay", "z^-1", "delay1"):
                ts = float(tokens[3]) if len(tokens) > 3 else 0.01
                self.dynamic_diagram.add_block(UnitDelayBlock(bname, sample_time=ts))
            elif btype in ("switch", "mux2to1"):
                thresh = float(tokens[3]) if len(tokens) > 3 else 0.0
                self.dynamic_diagram.add_block(SwitchBlock(bname, threshold=thresh))
            elif btype == "pid":
                kp = float(tokens[3]) if len(tokens) > 3 else 1.0
                ki = float(tokens[4]) if len(tokens) > 4 else 0.0
                kd = float(tokens[5]) if len(tokens) > 5 else 0.0
                n_filt = float(tokens[6]) if len(tokens) > 6 else 100.0
                lower = float(tokens[7]) if len(tokens) > 7 and tokens[7] != "none" else None
                upper = float(tokens[8]) if len(tokens) > 8 and tokens[8] != "none" else None
                self.dynamic_diagram.add_block(PIDBlock(bname, kp=kp, ki=ki, kd=kd, n_filter=n_filt, lower_limit=lower, upper_limit=upper))
            elif btype in ("const", "constant"):
                val = float(tokens[3]) if len(tokens) > 3 else 1.0
                self.dynamic_diagram.add_block(ConstantBlock(bname, value=val))
            elif btype == "step":
                step_t = float(tokens[3]) if len(tokens) > 3 else 0.0
                amp = float(tokens[4]) if len(tokens) > 4 else 1.0
                self.dynamic_diagram.add_block(StepSourceBlock(bname, step_time=step_t, amplitude=amp))
            elif btype in ("ramp", "slope"):
                slope = float(tokens[3]) if len(tokens) > 3 else 1.0
                start_t = float(tokens[4]) if len(tokens) > 4 else 0.0
                self.dynamic_diagram.add_block(RampSourceBlock(bname, slope=slope, start_time=start_t))
            elif btype in ("sine", "sin"):
                freq = float(tokens[3]) if len(tokens) > 3 else 1.0
                amp = float(tokens[4]) if len(tokens) > 4 else 1.0
                self.dynamic_diagram.add_block(SineSourceBlock(bname, freq=freq, amplitude=amp))
            elif btype in ("pulse", "square"):
                prd = float(tokens[3]) if len(tokens) > 3 else 1.0
                duty = float(tokens[4]) if len(tokens) > 4 else 0.5
                self.dynamic_diagram.add_block(PulseGeneratorBlock(bname, period=prd, duty_cycle=duty))
            elif btype in ("noise", "whitenoise"):
                pwr = float(tokens[3]) if len(tokens) > 3 else 0.1
                ts = float(tokens[4]) if len(tokens) > 4 else 0.01
                self.dynamic_diagram.add_block(BandLimitedWhiteNoiseBlock(bname, noise_power=pwr, sample_time=ts))
            elif btype == "scope":
                self.dynamic_diagram.add_block(ScopeSinkBlock(bname))
            elif btype == "tf":
                # add tf Plant [1] [1, 2, 1] or [1] [0.2 1]
                brackets = re.findall(r'\[([^\]]+)\]', line)
                if len(brackets) >= 2:
                    num_str, den_str = brackets[0], brackets[1]
                elif len(tokens) >= 5:
                    num_str = tokens[3].strip("[]")
                    den_str = tokens[4].strip("[]")
                else:
                    raise ValueError("Usage: add tf <name> [num_coeffs] [den_coeffs]")

                num = [float(x) for x in re.split(r'[\s,]+', num_str.strip()) if x]
                den = [float(x) for x in re.split(r'[\s,]+', den_str.strip()) if x]
                self.dynamic_diagram.add_block(TransferFunctionBlock(bname, num, den))
            else:
                raise ValueError(f"Unknown block type '{btype}'")

            return f"[green]Added Dynamic Block:[/green] {bname} ({btype})"

        if first == "connect":
            if len(tokens) < 3:
                raise ValueError("Usage: connect <BlockA.port> <BlockB.port>")
            self.dynamic_diagram.connect(tokens[1], tokens[2])
            return f"[green]Connected signals:[/green] {tokens[1]} -> {tokens[2]}"

        if first in ("sim", "simulate", "run"):
            t_stop = parse_eng_unit(tokens[1]) if len(tokens) > 1 else 10.0
            dt = parse_eng_unit(tokens[2]) if len(tokens) > 2 else 0.001
            res = self.dynamic_simulator.simulate(t_stop, dt=dt)
            self.last_dynamic_sim = res

            block_diagram = SchematicVisualizer.render_dynamic_block_diagram(
                self.dynamic_diagram.blocks,
                self.dynamic_diagram.connections
            )

            plots = []
            for sname, wf in res.items():
                p = AsciiPlotter.plot(wf.x, wf.y, title=f"Scope Output: {sname}", x_label="Time (s)")
                plots.append(p)

            return f"{block_diagram}\n\nSimulation complete ({t_stop}s).\n\n" + "\n\n".join(plots)

        raise ValueError(f"Unknown dynamic system command '{line}'")

    # --- Digital Logic Handlers ---
    def _handle_digital(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        if first == "truth" or first == "truthtable":
            expr = line[len(first):].strip()
            return HDLParser.generate_truth_table(expr)

        if first in ("wire", "gate", "dff", "clock", "input", "output"):
            circuit = HDLParser.parse(line)
            # Merge into current logic circuit
            self.logic_circuit.wires.update(circuit.wires)
            self.logic_circuit.gates.update(circuit.gates)
            self.logic_circuit.flip_flops.update(circuit.flip_flops)
            self.logic_circuit.clocks.update(circuit.clocks)
            return f"[green]Updated Digital Logic Circuit:[/green] {len(self.logic_circuit.gates)} gate(s), {len(self.logic_circuit.flip_flops)} FF(s)"

        if first in ("sim", "simulate", "run"):
            max_t = parse_eng_unit(tokens[1]) if len(tokens) > 1 else 100.0
            # If in seconds, convert to ns for digital simulator
            if max_t < 1e-3:
                max_t = max_t * 1e9
            sim = EventSimulator(self.logic_circuit)
            traces = sim.run(max_time_ns=max_t)
            self.last_logic_traces = traces
            return DigitalWaveformTracer.render_timing_diagram(traces, max_time_ns=max_t)

        raise ValueError(f"Unknown digital logic command '{line}'")

    # --- Embedded Handlers ---
    def _handle_embedded(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        if first == "load":
            asm_code = line[4:].strip()
            if os.path.exists(asm_code):
                with open(asm_code, "r", encoding="utf-8") as f:
                    asm_code = f.read()
            count = self.mcu.load_program(asm_code)
            return f"[green]Loaded MCU program:[/green] {count} instruction(s) assembled.\n\n" + Disassembler.disassemble(self.mcu.prog_mem, pc_highlight=0)

        if first == "step":
            ok = self.mcu.step()
            pc = self.mcu.regs.PC
            dis = Disassembler.disassemble(self.mcu.prog_mem, pc_highlight=pc)
            status = self.mcu.dump_state()
            return f"{status}\n\n{dis}"

        if first in ("run", "exec"):
            max_cyc = int(tokens[1]) if len(tokens) > 1 else 10000
            cycles = self.mcu.run(max_cycles=max_cyc)
            return f"Execution stopped after {cycles} cycles.\n\n" + self.mcu.dump_state()

        if first == "dump":
            return self.mcu.dump_state()

        if first == "reset":
            self.mcu.reset()
            return "MCU Reset complete."

        if first == "pwm":
            if len(tokens) < 3:
                raise ValueError("Usage: pwm <period_counts> <duty_counts>")
            prd = int(tokens[1])
            duty = int(tokens[2])
            self.mcu.peripherals.epwm.write(0x00, prd)
            self.mcu.peripherals.epwm.write(0x01, duty)
            wf = self.mcu.peripherals.epwm.generate_waveform(cycles=2000)
            plot_str = AsciiPlotter.plot(wf.x, wf.y, title=f"ePWM Output (Duty: {self.mcu.peripherals.epwm.duty_cycle*100:.1f}%)", x_label="Time (s)")
            return f"Configured ePWM1A: Period={prd}, Duty={duty}\n\n{plot_str}"

        raise ValueError(f"Unknown embedded command '{line}'")

    # --- Unified Handlers ---
    def _handle_unified(self, line: str, tokens: List[str]) -> str:
        # Tries to infer command type
        first = tokens[0].lower()
        if first in ("add", "connect", "run", ".ac", ".tran", ".op", ".dc"):
            return self._handle_circuit(line, tokens)
        return self._handle_numerical(line, tokens)


class TerminusApp(App):
    """The unified Textual TUI for TerminusECE."""

    CSS = """
    #command_input {
        dock: bottom;
        margin: 0 1;
        border: heavy cyan;
    }
    #console {
        height: 100%;
        margin: 0 1;
        border: solid green;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("c", "switch_circuit", "Circuit Mode"),
        ("n", "switch_numerical", "Numerical Mode"),
        ("s", "switch_dynamic", "Dynamic Systems"),
        ("l", "switch_digital", "Digital Logic"),
        ("e", "switch_embedded", "Embedded MCU"),
        ("q", "quit_app", "Quit Terminus"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bridge = TerminusEngineBridge()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="console", highlight=True, markup=True)
        yield Input(placeholder="Terminus [CIRCUIT] > Enter command (e.g. 'help', 'add V1 10V', 'mode numerical')...", id="command_input")
        yield Footer()

    def on_ready(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]=== TerminusECE Unified Workbench Initialized ===[/bold green]")
        log.write("Lightning-fast command-driven EDA workspace for Electrical & Computer Engineering.")
        log.write("Current mode: [bold magenta]CIRCUIT (LTspice)[/bold magenta]. Type [bold cyan]'help'[/bold cyan] for commands.")
        self.query_one(Input).focus()

    def action_quit_app(self) -> None:
        self.exit()

    def action_switch_circuit(self) -> None:
        self._set_mode("CIRCUIT")

    def action_switch_numerical(self) -> None:
        self._set_mode("NUMERICAL")

    def action_switch_dynamic(self) -> None:
        self._set_mode("DYNAMIC")

    def action_switch_digital(self) -> None:
        self._set_mode("DIGITAL")

    def action_switch_embedded(self) -> None:
        self._set_mode("EMBEDDED")

    def _set_mode(self, mode: str) -> None:
        new_mode = self.bridge.switch_mode(mode)
        inp = self.query_one(Input)
        inp.placeholder = f"Terminus [{new_mode}] > Enter command..."
        log = self.query_one(RichLog)
        log.write(f"[bold magenta]Switched context to {new_mode}[/bold magenta]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        cmd = event.value.strip()
        if not cmd:
            return

        log.write(f"[bold cyan]> {cmd}[/bold cyan]")
        try:
            out = self.bridge.execute_command(cmd)
            if out:
                log.write(out)
        except Exception as err:
            log.write(f"[bold red]Error:[/bold red] {err}")

        # Update input placeholder if mode changed
        self.query_one(Input).placeholder = f"Terminus [{self.bridge.mode}] > Enter command..."
        event.input.value = ""
