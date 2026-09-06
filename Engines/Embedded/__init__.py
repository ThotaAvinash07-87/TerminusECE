"""Embedded Microcontroller Emulator and Peripheral Engine (C2000 / DSP Architecture)."""

from .mcu_core import MCUCore, RegisterFile, CPUFlags
from .toolchain import Assembler, Disassembler, Instruction
from .peripherals import PeripheralBus, GPIOController, EPWMGenerator, TimerModule, ADCSampler

__all__ = [
    "MCUCore",
    "RegisterFile",
    "CPUFlags",
    "Assembler",
    "Disassembler",
    "Instruction",
    "PeripheralBus",
    "GPIOController",
    "EPWMGenerator",
    "TimerModule",
    "ADCSampler",
]
