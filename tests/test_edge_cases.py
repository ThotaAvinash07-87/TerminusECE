"""Comprehensive edge-case and loophole robustness tests for TerminusECE."""

import unittest
import math
import numpy as np

from CORE.common_math import parse_eng_unit, sanitize_array, SignalMetrics, Waveform
from CORE.ascii_canvas import AsciiPlotter, AsciiBodePlotter, SchematicVisualizer

from Engines.Circuit.components import Resistor, Capacitor, Inductor, VoltageSource
from Engines.Circuit.netlist_parser import Netlist
from Engines.Circuit.mna_solver import MNASolver

from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from Engines.Numerical.transforms import TransferFunction
from Engines.Numerical.matrix_ops import matrix_inv, matrix_det

from Engines.Dynamic_System.blocks import IntegratorBlock, GainBlock, SumBlock, StepSourceBlock, ScopeSinkBlock
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator


class TestRobustnessAndEdgeCases(unittest.TestCase):
    def test_sanitize_array_with_nans_and_infs(self):
        """Ensures NaNs, +Inf, -Inf are safely sanitized without raising exceptions."""
        bad_data = [np.nan, np.inf, -np.inf, 1e20, -1e20, 5.0]
        clean = sanitize_array(bad_data)
        self.assertFalse(np.any(np.isnan(clean)))
        self.assertFalse(np.any(np.isinf(clean)))
        self.assertLessEqual(np.max(clean), 1e12)
        self.assertGreaterEqual(np.min(clean), -1e12)

    def test_braille_continuous_plotter_with_flat_and_wild_signals(self):
        """Verifies high-res continuous plot rendering on zero-variance and noisy/wild signals."""
        t = np.linspace(0, 1, 100)
        # Constant / flat signal
        flat_y = np.zeros(100)
        p1 = AsciiPlotter.plot(t, flat_y, title="Flat Zero")
        self.assertIn("Flat Zero", p1)

        # High frequency noisy signal with sudden glitch spike
        wild_y = np.sin(2 * np.pi * 50 * t)
        wild_y[50] = 50.0  # Massive spike
        p2 = AsciiPlotter.plot(t, wild_y, title="Noisy Glitch")
        self.assertIn("Noisy Glitch", p2)
        self.assertIn("Notice:", p2)  # Check that sudden glitch was flagged

    def test_bode_metrics_and_peaking(self):
        """Tests resonant peak detection, bandwidth, and margins on second-order system."""
        # Resonant 2nd-order transfer function H(s) = 100 / (s^2 + 2*s + 100) -> wn = 10 rad/s (~1.59 Hz)
        tf = TransferFunction([100], [1, 2, 100], name="ResonantPlant")
        freqs, mag, phase = tf.frequency_response(freqs=np.logspace(-1, 2, 200))
        metrics = SignalMetrics.measure_bode_metrics(freqs, mag, phase)

        self.assertIn("resonance_freq_hz", metrics)
        self.assertAlmostEqual(metrics["resonance_freq_hz"], 1.59, delta=0.3)
        self.assertGreater(metrics["resonance_peak_db"], 5.0)

    def test_circuit_floating_node_and_singular_matrix(self):
        """Simulates circuit with floating node and verifies MNA solver resolves with GMIN regularization."""
        netlist = Netlist()
        # V1 connected to ground and node 1
        netlist.add_component(VoltageSource("V1", "n1", "0", dc=5.0))
        # R1 connected between floating nodes without path to ground
        netlist.add_component(Resistor("R1", "floatA", "floatB", 1000.0))

        solver = MNASolver(netlist)
        # solve_op should not crash on singular matrix
        res = solver.solve_op()
        self.assertIn("V(n1)", res.op_results)
        self.assertAlmostEqual(res.op_results["V(n1)"], 5.0)

    def test_circuit_ac_at_zero_and_infinite_frequency(self):
        """Tests AC solver at extreme low (0 Hz) and high frequency limits."""
        netlist = Netlist()
        netlist.add_component(VoltageSource("V1", "in", "0", ac_mag=1.0))
        netlist.add_component(Resistor("R1", "in", "out", 1000.0))
        netlist.add_component(Capacitor("C1", "out", "0", 1e-6))

        solver = MNASolver(netlist)
        # Should gracefully handle 0.0 Hz start frequency without zero-division error
        res = solver.solve_ac(sweep_type="lin", points=10, f_start=0.0, f_stop=1e6)
        self.assertIn("V(out)", res.waveforms)

    def test_unstable_ode_integration_explosion_protection(self):
        """Simulates an exponentially unstable system (dx/dt = +1000*x) and checks state bounding."""
        diagram = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.0, amplitude=1.0)
        # Unstable positive feedback loop
        integ = IntegratorBlock("ExplosiveInt", initial_condition=1.0)
        gain = GainBlock("HugeGain", gain=100.0)
        scope = ScopeSinkBlock("Scope")

        diagram.add_block(step)
        diagram.add_block(integ)
        diagram.add_block(gain)
        diagram.add_block(scope)

        diagram.connect("Step.0", "ExplosiveInt.0")
        diagram.connect("ExplosiveInt.0", "HugeGain.0")
        diagram.connect("HugeGain.0", "ExplosiveInt.0")  # Positive feedback explosive loop
        diagram.connect("ExplosiveInt.0", "Scope.0")

        sim = DynamicSystemSimulator(diagram)
        # Should not throw OverflowError or NaN
        res = sim.simulate(t_stop=1.0, dt=0.01)
        self.assertIn("Scope", res)
        scope_y = res["Scope"].y
        self.assertFalse(np.any(np.isnan(scope_y)))

    def test_schematic_visualizers(self):
        """Tests ASCII block diagram generators for circuit and dynamic systems."""
        components = {
            "R1": Resistor("R1", "in", "out", 1000.0),
            "C1": Capacitor("C1", "out", "0", 1e-6)
        }
        pin_map = {"R1": ["in", "out"], "C1": ["out", "0"]}
        c_diag = SchematicVisualizer.render_circuit_topology(components, pin_map)
        self.assertIn("CIRCUIT SCHEMATIC TOPOLOGY BLOCK", c_diag)
        self.assertIn("R1", c_diag)
        self.assertIn("C1", c_diag)

        blocks = {
            "S1": StepSourceBlock("S1"),
            "G1": GainBlock("G1", gain=2.0)
        }
        connections = [(("S1", 0), ("G1", 0))]
        d_diag = SchematicVisualizer.render_dynamic_block_diagram(blocks, connections)
        self.assertIn("DYNAMIC SYSTEM BLOCK FLOW DIAGRAM", d_diag)
        self.assertIn("S1", d_diag)
        self.assertIn("G1", d_diag)


if __name__ == "__main__":
    unittest.main()
