"""Comprehensive unit and integration test suite for TerminusECE."""

import unittest
import math
import numpy as np

from CORE.common_math import parse_eng_unit, format_eng_unit, Waveform, SignalMetrics
from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter
from CORE.ipc_router import IPCRouter, IPCClient

from Engines.Circuit.components import Resistor, Capacitor, VoltageSource
from Engines.Circuit.netlist_parser import Netlist, CircuitParser
from Engines.Circuit.mna_solver import MNASolver

from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from Engines.Numerical.transforms import TransferFunction
from Engines.Numerical.matrix_ops import matrix_det, matrix_inv, matrix_eig

from Engines.Dynamic_System.blocks import IntegratorBlock, GainBlock, SumBlock, StepSourceBlock, ScopeSinkBlock
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator

from Engines.Digital_Logic.gates import LogicValue, AndGate, XorGate, DFlipFlop
from Engines.Digital_Logic.hdl_parser import HDLParser
from Engines.Digital_Logic.event_sim import EventSimulator, DigitalWaveformTracer

from Engines.Embedded.mcu_core import MCUCore
from Engines.Embedded.toolchain import Assembler, Disassembler
from UI.app import TerminusEngineBridge


class TestCoreUtilities(unittest.TestCase):
    def test_unit_parser(self):
        self.assertAlmostEqual(parse_eng_unit("10k"), 10000.0)
        self.assertAlmostEqual(parse_eng_unit("1u"), 1e-6)
        self.assertAlmostEqual(parse_eng_unit("100kHz"), 100000.0)
        self.assertAlmostEqual(parse_eng_unit("2.2MEG"), 2200000.0)
        self.assertAlmostEqual(parse_eng_unit("5mV"), 0.005)
        self.assertAlmostEqual(parse_eng_unit("10pF"), 1e-11)
        self.assertAlmostEqual(parse_eng_unit("100"), 100.0)

    def test_ascii_canvas_routing(self):
        canvas = AsciiCanvas(width=40, height=15)
        canvas.draw_box(2, 2, 8, 4, title="R1")
        canvas.draw_box(25, 2, 8, 4, title="C1")
        path = canvas.route_wire((10, 4), (25, 4))
        self.assertGreater(len(path), 0)
        rendered = canvas.render()
        self.assertIn("R1", rendered)
        self.assertIn("C1", rendered)

    def test_ascii_plotter(self):
        t = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * t)
        plot_str = AsciiPlotter.plot(t, y, width=40, height=8, title="Sine Wave")
        self.assertIn("Sine Wave", plot_str)
        self.assertIn("┌", plot_str)


class TestCircuitEngine(unittest.TestCase):
    def test_rc_lowpass_filter_ac(self):
        """Simulates an RC filter (R=1k, C=1uF -> fc = 1/(2*pi*R*C) ~= 159.15 Hz)."""
        netlist = Netlist()
        # V1 in 0 AC 1
        netlist.add_component(VoltageSource("V1", "in", "0", ac_mag=1.0))
        # R1 in out 1k
        netlist.add_component(Resistor("R1", "in", "out", 1000.0))
        # C1 out 0 1u
        netlist.add_component(Capacitor("C1", "out", "0", 1e-6))

        solver = MNASolver(netlist)
        res = solver.solve_ac(sweep_type="dec", points=50, f_start=1.0, f_stop=10000.0)
        self.assertIn("V(out)", res.waveforms)

        out_wf = res.waveforms["V(out)"]
        fc = SignalMetrics.measure_cutoff_frequency(out_wf.x, out_wf.magnitude_db)
        self.assertIsNotNone(fc)
        # Expected cutoff: ~159.15 Hz
        self.assertAlmostEqual(fc, 159.15, delta=15.0)

    def test_dc_voltage_divider(self):
        """Tests DC operating point of a 2:1 resistive divider."""
        netlist = Netlist()
        netlist.add_component(VoltageSource("V1", "n1", "0", dc=10.0))
        netlist.add_component(Resistor("R1", "n1", "n2", 1000.0))
        netlist.add_component(Resistor("R2", "n2", "0", 1000.0))

        solver = MNASolver(netlist)
        op = solver.solve_op()
        self.assertAlmostEqual(op.op_results["V(n1)"], 10.0, places=4)
        self.assertAlmostEqual(op.op_results["V(n2)"], 5.0, places=4)


class TestNumericalEngine(unittest.TestCase):
    def test_matrix_operations(self):
        ws = NumericalWorkspace()
        parser = NumericalASTParser(ws)
        parser.execute("A = [1 2; 3 4]")
        parser.execute("b = [5; 11]")
        parser.execute("x = inv(A) @ b")

        x_val = ws.variables["x"]
        # Expected x = [1; 2]
        self.assertAlmostEqual(float(x_val[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(x_val[1, 0]), 2.0, places=4)

    def test_transfer_function(self):
        tf = TransferFunction([1], [1, 2, 1], name="H(s)")
        self.assertTrue(tf.is_stable())
        self.assertEqual(len(tf.poles), 2)
        step_wf = tf.step_response(t=np.linspace(0, 5, 50))
        self.assertAlmostEqual(step_wf.y[-1], 1.0, delta=0.05)


class TestDynamicSystemsEngine(unittest.TestCase):
    def test_closed_loop_feedback(self):
        """Step -> (Sum) -> Integrator (1/s) -> Scope with feedback."""
        diagram = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.0, amplitude=1.0)
        sum_blk = SumBlock("Sum", signs="+-")
        integ = IntegratorBlock("Plant", initial_condition=0.0)
        scope = ScopeSinkBlock("Scope")

        diagram.add_block(step)
        diagram.add_block(sum_blk)
        diagram.add_block(integ)
        diagram.add_block(scope)

        # Connect: Step.0 -> Sum.0
        diagram.connect("Step.0", "Sum.0")
        # Sum.0 -> Plant.0
        diagram.connect("Sum.0", "Plant.0")
        # Plant.0 -> Scope.0
        diagram.connect("Plant.0", "Scope.0")
        # Feedback: Plant.0 -> Sum.1
        diagram.connect("Plant.0", "Sum.1")

        sim = DynamicSystemSimulator(diagram)
        results = sim.simulate(t_stop=5.0, dt=0.01)

        self.assertIn("Scope", results)
        scope_wf = results["Scope"]
        # Step response of 1/(s+1) approaches 1.0
        self.assertAlmostEqual(scope_wf.y[-1], 1.0, delta=0.05)


class TestDigitalLogicEngine(unittest.TestCase):
    def test_half_adder_and_truth_table(self):
        hdl = """
        input A, B
        output Sum, Carry
        gate G1 XOR A B -> Sum
        gate G2 AND A B -> Carry
        """
        circuit = HDLParser.parse(hdl)
        self.assertEqual(len(circuit.gates), 2)

        tt = HDLParser.generate_truth_table("A ^ B", ["A", "B"])
        self.assertIn("Result", tt)

    def test_d_flip_flop(self):
        circuit = HDLParser.parse("""
        wire d, clk, q
        clock CLK period=10ns
        dff D1 clk=CLK d=d q=q
        """)
        sim = EventSimulator(circuit)
        sim.schedule_event(0.0, "d", LogicValue.HIGH)
        traces = sim.run(max_time_ns=30.0)
        self.assertIn("q", traces)
        # q should become HIGH after rising edge of CLK
        final_q = traces["q"][-1][1]
        self.assertEqual(final_q, LogicValue.HIGH)


class TestEmbeddedEngine(unittest.TestCase):
    def test_asm_loop_execution(self):
        """Calculates 1 + 2 + 3 + 4 + 5 = 15."""
        asm = """
        MOV R0, #5
        MOV R1, #0
        LOOP:
        ADD R1, R0
        SUB R0, #1
        CMP R0, #0
        JNZ LOOP
        HALT
        """
        mcu = MCUCore()
        mcu.load_program(asm)
        cycles = mcu.run(max_cycles=100)
        self.assertEqual(mcu.regs.R[1], 15)
        self.assertTrue(mcu.regs.halted)


class TestTerminusEngineBridge(unittest.TestCase):
    def test_cli_circuit_pipeline(self):
        bridge = TerminusEngineBridge()
        out1 = bridge.execute_command("mode circuit")
        self.assertIn("CIRCUIT", out1)
        bridge.execute_command("add V1 10V ac=1")
        bridge.execute_command("add R1 1k")
        bridge.execute_command("add C1 1u")
        bridge.execute_command("connect V1.p | R1.a")
        bridge.execute_command("connect R1.b | C1.a | node_out")
        bridge.execute_command("connect V1.n | C1.b | gnd")
        sim_out = bridge.execute_command("run .ac dec 10 1Hz 100kHz")
        self.assertIn("Bode Plot", sim_out)


if __name__ == "__main__":
    unittest.main()
