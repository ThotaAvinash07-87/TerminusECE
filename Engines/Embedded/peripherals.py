"""Memory-mapped peripherals: GPIO, ePWM, Timers, and ADC for MCU emulation."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from CORE.common_math import Waveform


class GPIOController:
    """General Purpose Input/Output controller with Direction and Data registers."""

    def __init__(self, base_addr: int = 0x7000):
        self.base_addr = base_addr
        # 0x00: GPADIR, 0x01: GPADAT, 0x02: GPASET, 0x03: GPACLEAR
        self.dir: int = 0x00000000  # 1 = output, 0 = input
        self.data: int = 0x00000000

    def write(self, offset: int, value: int) -> None:
        if offset == 0x00:  # GPADIR
            self.dir = value & 0xFFFFFFFF
        elif offset == 0x01:  # GPADAT
            self.data = (value & self.dir) | (self.data & ~self.dir)
        elif offset == 0x02:  # GPASET
            self.data |= (value & self.dir)
        elif offset == 0x03:  # GPACLEAR
            self.data &= ~(value & self.dir)

    def read(self, offset: int) -> int:
        if offset == 0x00:
            return self.dir
        elif offset == 0x01:
            return self.data
        return 0


class EPWMGenerator:
    """Enhanced Pulse Width Modulator (ePWM) peripheral."""

    def __init__(self, base_addr: int = 0x7100):
        self.base_addr = base_addr
        # 0x00: TBPRD (Time Base Period), 0x01: CMPA (Compare A), 0x02: TBCTL (Control)
        self.tbprd: int = 1000  # Period in clock counts
        self.cmpa: int = 500    # Duty threshold
        self.tbctl: int = 0x0001 # 1 = Enabled
        self.counter: int = 0

    @property
    def duty_cycle(self) -> float:
        if self.tbprd <= 0:
            return 0.0
        return float(min(1.0, max(0.0, self.cmpa / self.tbprd)))

    def write(self, offset: int, value: int) -> None:
        if offset == 0x00:
            self.tbprd = max(1, value & 0xFFFF)
        elif offset == 0x01:
            self.cmpa = value & 0xFFFF
        elif offset == 0x02:
            self.tbctl = value & 0xFFFF

    def read(self, offset: int) -> int:
        if offset == 0x00:
            return self.tbprd
        elif offset == 0x01:
            return self.cmpa
        elif offset == 0x02:
            return self.tbctl
        return 0

    def generate_waveform(self, cycles: int = 5000, dt_ns: float = 10.0) -> Waveform:
        """Generates continuous time-series PWM signal trace."""
        period_counts = self.tbprd
        duty_counts = self.cmpa
        t = np.arange(cycles) * (dt_ns * 1e-9)
        sig = np.zeros(cycles)

        for i in range(cycles):
            count_in_period = i % period_counts
            sig[i] = 3.3 if count_in_period < duty_counts else 0.0

        return Waveform(t, sig, name="ePWM1A", x_unit="s", y_unit="V", domain="time")


class TimerModule:
    """32-bit CPU countdown timer."""

    def __init__(self, base_addr: int = 0x7200):
        self.base_addr = base_addr
        self.period: int = 10000
        self.counter: int = 10000
        self.control: int = 0x0001
        self.interrupt_flag: bool = False

    def tick(self) -> bool:
        """Decrements timer and triggers interrupt on underflow."""
        if self.control & 0x0001:
            self.counter -= 1
            if self.counter <= 0:
                self.counter = self.period
                self.interrupt_flag = True
                return True
        return False

    def write(self, offset: int, value: int) -> None:
        if offset == 0x00:
            self.period = value & 0xFFFFFFFF
            self.counter = self.period
        elif offset == 0x01:
            self.control = value & 0xFFFF

    def read(self, offset: int) -> int:
        if offset == 0x00:
            return self.period
        elif offset == 0x01:
            return self.counter
        elif offset == 0x02:
            return self.control
        return 0


class ADCSampler:
    """Analog to Digital Converter emulator."""

    def __init__(self, base_addr: int = 0x7300):
        self.base_addr = base_addr
        self.channels: List[float] = [0.0] * 8  # 8 analog channels (0.0 to 3.3V)
        self.results: List[int] = [0] * 8       # 12-bit digital codes (0 to 4095)

    def set_analog_input(self, channel: int, voltage: float) -> None:
        if 0 <= channel < 8:
            v_clamped = max(0.0, min(3.3, voltage))
            self.channels[channel] = v_clamped
            self.results[channel] = int((v_clamped / 3.3) * 4095)

    def read(self, offset: int) -> int:
        if 0 <= offset < 8:
            return self.results[offset]
        return 0


class PeripheralBus:
    """Memory-mapped peripheral address router."""

    def __init__(self):
        self.gpio = GPIOController(0x7000)
        self.epwm = EPWMGenerator(0x7100)
        self.timer = TimerModule(0x7200)
        self.adc = ADCSampler(0x7300)

    def is_peripheral_addr(self, addr: int) -> bool:
        return 0x7000 <= addr < 0x8000

    def write(self, addr: int, value: int) -> None:
        base = addr & 0xFF00
        offset = addr & 0x00FF

        if base == 0x7000:
            self.gpio.write(offset, value)
        elif base == 0x7100:
            self.epwm.write(offset, value)
        elif base == 0x7200:
            self.timer.write(offset, value)

    def read(self, addr: int) -> int:
        base = addr & 0xFF00
        offset = addr & 0x00FF

        if base == 0x7000:
            return self.gpio.read(offset)
        elif base == 0x7100:
            return self.epwm.read(offset)
        elif base == 0x7200:
            return self.timer.read(offset)
        elif base == 0x7300:
            return self.adc.read(offset)
        return 0
