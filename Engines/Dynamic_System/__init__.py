"""Dynamic Systems and Control Simulation Engine (Simulink-like block diagram & ODE solver)."""

from .blocks import (
    Block,
    IntegratorBlock,
    GainBlock,
    SumBlock,
    PIDBlock,
    TransferFunctionBlock,
    StepSourceBlock,
    SineSourceBlock,
    SaturationBlock,
    ScopeSinkBlock,
)
from .scheduler import SystemDiagram, Scheduler
from .ode_solver import DynamicSystemSimulator

__all__ = [
    "Block",
    "IntegratorBlock",
    "GainBlock",
    "SumBlock",
    "PIDBlock",
    "TransferFunctionBlock",
    "StepSourceBlock",
    "SineSourceBlock",
    "SaturationBlock",
    "ScopeSinkBlock",
    "SystemDiagram",
    "Scheduler",
    "DynamicSystemSimulator",
]
