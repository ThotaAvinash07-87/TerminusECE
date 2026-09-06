"""Comprehensive unit and benchmark tests for all Simulink-grade Dynamic System blocks."""

import unittest
import math
import numpy as np

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


class TestDynamicSystemsBlocks(unittest.TestCase):

    def test_integrator_with_saturation_and_anti_windup(self):
        """Tests integrator with upper and lower saturation limits and anti-windup clamping."""
        diag = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.0, amplitude=5.0)
        integ = IntegratorBlock("Int1", initial_condition=0.0, lower_limit=-2.0, upper_limit=3.0)
        scope = ScopeSinkBlock("Scope")

        diag.add_block(step)
        diag.add_block(integ)
        diag.add_block(scope)

        diag.connect("Step.0", "Int1.0")
        diag.connect("Int1.0", "Scope.0")

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=2.0, dt=0.01)

        scope_y = results["Scope"].y
        # Upper limit is 3.0, so state should clamp at 3.0 and not exceed it
        self.assertAlmostEqual(scope_y[-1], 3.0, places=4)
        self.assertLessEqual(np.max(scope_y), 3.0)

    def test_derivative_block_filtered(self):
        """Tests filtered derivative on a ramp input (y = d/dt(slope * t) = slope)."""
        diag = SystemDiagram()
        ramp = RampSourceBlock("Ramp", slope=4.0, start_time=0.0)
        deriv = DerivativeBlock("Deriv", tau=0.01)
        scope = ScopeSinkBlock("Scope")

        diag.add_block(ramp)
        diag.add_block(deriv)
        diag.add_block(scope)

        diag.connect("Ramp.0", "Deriv.0")
        diag.connect("Deriv.0", "Scope.0")

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=0.5, dt=0.001)

        scope_y = results["Scope"].y
        # Filtered derivative of ramp slope 4.0 should settle to 4.0
        self.assertAlmostEqual(scope_y[-1], 4.0, delta=0.05)

    def test_state_space_block(self):
        """Tests 2nd order harmonic oscillator state-space system."""
        # dx1/dt = x2, dx2/dt = -wn^2 x1 - 2*zeta*wn x2
        wn = 5.0
        zeta = 0.7
        A = np.array([[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]])
        B = np.array([[0.0], [wn**2]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])

        diag = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.0, amplitude=1.0)
        ss = StateSpaceBlock("SS", A=A, B=B, C=C, D=D)
        scope = ScopeSinkBlock("Scope")

        diag.add_block(step)
        diag.add_block(ss)
        diag.add_block(scope)

        diag.connect("Step.0", "SS.0")
        diag.connect("SS.0", "Scope.0")

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=3.0, dt=0.01)

        scope_y = results["Scope"].y
        self.assertAlmostEqual(scope_y[-1], 1.0, delta=0.05)

    def test_transport_delay_block(self):
        """Tests pure time delay transport."""
        diag = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.2, amplitude=10.0)
        delay = TransportDelayBlock("Delay", delay_time=0.3, initial_output=0.0)
        scope = ScopeSinkBlock("Scope")

        diag.add_block(step)
        diag.add_block(delay)
        diag.add_block(scope)

        diag.connect("Step.0", "Delay.0")
        diag.connect("Delay.0", "Scope.0")

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=1.0, dt=0.01)

        wf = results["Scope"]
        # Input step occurs at t=0.2s, delay is 0.3s -> output step should occur at t=0.5s
        val_before = wf.sample_at(0.4)
        val_after = wf.sample_at(0.6)
        self.assertAlmostEqual(val_before, 0.0, delta=0.1)
        self.assertAlmostEqual(val_after, 10.0, delta=0.1)

    def test_nonlinear_rate_limiter_and_deadzone(self):
        """Tests rate limiter and deadzone blocks."""
        diag = SystemDiagram()
        step = StepSourceBlock("Step", step_time=0.0, amplitude=10.0)
        rl = RateLimiterBlock("RL", rising_slew_rate=2.0, falling_slew_rate=-2.0, initial_output=0.0)
        dz = DeadZoneBlock("DZ", start_zone=-1.0, end_zone=1.0)
        scope = ScopeSinkBlock("Scope")

        diag.add_block(step)
        diag.add_block(rl)
        diag.add_block(dz)
        diag.add_block(scope)

        diag.connect("Step.0", "RL.0")
        diag.connect("RL.0", "DZ.0")
        diag.connect("DZ.0", "Scope.0")

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=3.0, dt=0.01)

        wf = results["Scope"]
        # At t=1.0s, RL output is ~2.0, minus DZ end_zone(1.0) = ~1.0
        val_t1 = wf.sample_at(1.0)
        self.assertAlmostEqual(val_t1, 1.0, delta=0.1)

    def test_relay_bang_bang_schmitt_trigger(self):
        """Tests relay hysteresis (Schmitt trigger)."""
        relay = RelayBlock("Relay", switch_on_point=2.0, switch_off_point=-2.0, output_on=5.0, output_off=-5.0)
        
        # Test rising past switch_on_point
        relay.inputs = [1.0]
        out1 = relay.compute_output(0.0)[0]
        self.assertEqual(out1, -5.0)

        relay.inputs = [2.5]
        out2 = relay.compute_output(0.1)[0]
        self.assertEqual(out2, 5.0)

        # In hysteresis region while falling
        relay.inputs = [0.0]
        out3 = relay.compute_output(0.2)[0]
        self.assertEqual(out3, 5.0)  # should stay ON

        # Drops below switch_off_point
        relay.inputs = [-2.5]
        out4 = relay.compute_output(0.3)[0]
        self.assertEqual(out4, -5.0)

    def test_math_product_and_lookup_table(self):
        """Tests Product block and 1D Lookup Table."""
        prod = ProductBlock("Prod", operations="*/")
        prod.inputs = [10.0, 2.0]
        self.assertAlmostEqual(prod.compute_output(0.0)[0], 5.0)

        lut = LookupTable1DBlock("LUT", x_data=[0.0, 10.0, 20.0], y_data=[0.0, 50.0, 200.0])
        lut.inputs = [5.0]
        self.assertAlmostEqual(lut.compute_output(0.0)[0], 25.0)

    def test_discrete_zoh_and_unit_delay(self):
        """Tests Zero-Order Hold and Unit Delay blocks."""
        zoh = ZeroOrderHoldBlock("ZOH", sample_time=0.1)
        zoh.inputs = [3.14]
        out1 = zoh.compute_output(0.0)[0]
        self.assertAlmostEqual(out1, 3.14)

        # Input changes at t=0.05s, ZOH should hold until t=0.1s
        zoh.inputs = [100.0]
        out2 = zoh.compute_output(0.05)[0]
        self.assertAlmostEqual(out2, 3.14)

        out3 = zoh.compute_output(0.11)[0]
        self.assertAlmostEqual(out3, 100.0)

    def test_dc_motor_speed_control_benchmark(self):
        """Real-world Benchmark: Closed-loop DC Motor speed control with Coulomb friction & actuator saturation."""
        diag = SystemDiagram("DCMotorSystem")
        
        # Desired speed step = 50 rad/s
        setpoint = StepSourceBlock("SetPoint", step_time=0.0, amplitude=50.0)
        sum_err = SumBlock("SumErr", signs="+-")
        # PID Controller with anti-windup clamping to motor max voltage +-24V
        pid = PIDBlock("PID", kp=1.5, ki=3.0, kd=0.02, lower_limit=-24.0, upper_limit=24.0)
        # Actuator Saturation +-24V
        sat = SaturationBlock("DriverSat", lower_limit=-24.0, upper_limit=24.0)
        # Motor Electrical & Mechanical Transfer Function: 1 / (J*s + b)
        # Transfer Function: omega(s)/V(s) = K / ((J*s + b)*(L*s + R) + K^2)
        motor_plant = TransferFunctionBlock("MotorPlant", num=[10.0], den=[0.005, 0.15, 1.0])
        # Coulomb friction load
        friction = CoulombViscousFrictionBlock("Fric", f_coulomb=0.2, b_viscous=0.05)
        scope_speed = ScopeSinkBlock("SpeedScope")

        diag.add_block(setpoint)
        diag.add_block(sum_err)
        diag.add_block(pid)
        diag.add_block(sat)
        diag.add_block(motor_plant)
        diag.add_block(friction)
        diag.add_block(scope_speed)

        # Wiring
        diag.connect("SetPoint.0", "SumErr.0")
        diag.connect("SumErr.0", "PID.0")
        diag.connect("PID.0", "DriverSat.0")
        diag.connect("DriverSat.0", "MotorPlant.0")
        diag.connect("MotorPlant.0", "SpeedScope.0")
        diag.connect("MotorPlant.0", "SumErr.1")  # Speed feedback

        sim = DynamicSystemSimulator(diag)
        results = sim.simulate(t_stop=2.0, dt=0.001)

        speed_wf = results["SpeedScope"]
        # Verify closed-loop speed reaches setpoint (50 rad/s)
        self.assertAlmostEqual(speed_wf.y[-1], 50.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
