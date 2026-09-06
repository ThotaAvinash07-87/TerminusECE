"""Subsystem layout screens for TerminusECE."""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from .widgets import (
    AsciiPlotWidget,
    SchematicCanvasWidget,
    McuStateWidget,
    LogicTimingWidget,
)


class BaseSubsystemScreen(Screen):
    """Base layout screen containing split terminal console and visualization panel."""

    def __init__(self, mode_name: str, **kwargs):
        super().__init__(**kwargs)
        self.mode_name = mode_name.upper()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_container"):
            with Vertical(id="left_pane"):
                yield RichLog(id="console", highlight=True, markup=True)
            with Vertical(id="right_pane"):
                yield Static(f"[bold cyan]=== {self.mode_name} WORKSPACE ===[/bold cyan]", id="panel_header")
        yield Input(placeholder=f"Terminus [{self.mode_name}] > Enter command...", id="command_input")
        yield Footer()


class WorkbenchScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("UNIFIED WORKBENCH", **kwargs)


class CircuitScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("CIRCUIT (LTSPICE)", **kwargs)


class NumericalScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("NUMERICAL (MATLAB)", **kwargs)


class DynamicSystemScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("DYNAMIC SYSTEMS (SIMULINK)", **kwargs)


class DigitalLogicScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("DIGITAL LOGIC (XILINX)", **kwargs)


class EmbeddedScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("EMBEDDED (MCU/DSP)", **kwargs)
