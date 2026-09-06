"""Custom Textual widgets for plots, schematics, MCU debugger, and logic timing."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from textual.widgets import Static
from rich.text import Text
from rich.panel import Panel

from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter
from CORE.common_math import Waveform


class AsciiPlotWidget(Static):
    """Widget that renders high-contrast terminal waveform and frequency response plots."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]No plot data available.[/italic dim]", **kwargs)

    def set_waveform(self, wf: Waveform, title: str = "") -> None:
        plot_title = title or wf.name
        if wf.is_complex or wf.domain == "frequency":
            bode_str = AsciiBodePlotter.plot_bode(
                wf.x,
                wf.magnitude_db,
                wf.phase_deg,
                width=72,
                height_each=7,
                title=plot_title
            )
            self.update(Panel(Text.from_markup(bode_str), title="Frequency Response"))
        else:
            plot_str = AsciiPlotter.plot(
                wf.x,
                wf.y,
                width=72,
                height=14,
                title=plot_title,
                x_label=f"Time ({wf.x_unit})",
                y_label=f"Signal ({wf.y_unit})"
            )
            self.update(Panel(Text.from_markup(plot_str), title="Waveform Viewer"))


class SchematicCanvasWidget(Static):
    """Widget that renders ASCII 2D circuit schematics and dynamic block diagrams."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]Canvas empty.[/italic dim]", **kwargs)

    def render_circuit(self, netlist_components: Dict[str, Any], pin_map: Dict[str, List[str]]) -> None:
        canvas = AsciiCanvas(width=72, height=18)
        canvas.draw_box(0, 0, 72, 18, title="Circuit Schematic Canvas")

        col = 3
        row = 3
        for i, (cname, comp) in enumerate(netlist_components.items()):
            pins = pin_map.get(cname, comp.nodes)
            pins_str = ",".join(pins)
            val_str = str(getattr(comp, "value", getattr(comp, "dc", "")))
            
            # Place block
            canvas.draw_box(col, row, 18, 4, title=cname)
            canvas.draw_text(col + 2, row + 1, f"Val: {val_str[:8]}")
            canvas.draw_text(col + 2, row + 2, f"Nets: {pins_str[:8]}")

            col += 22
            if col > 50:
                col = 3
                row += 5

        self.update(Panel(Text(canvas.render()), title="Schematic Canvas"))


class McuStateWidget(Static):
    """Widget displaying register contents, flags, and memory dump of the embedded MCU."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]MCU not loaded.[/italic dim]", **kwargs)

    def update_state(self, state_text: str) -> None:
        self.update(Panel(Text.from_markup(state_text), title="MCU Core Debugger"))


class LogicTimingWidget(Static):
    """Widget displaying multi-channel digital logic trace timing diagrams."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]No logic simulation trace recorded.[/italic dim]", **kwargs)

    def set_timing_diagram(self, timing_str: str) -> None:
        self.update(Panel(Text.from_markup(timing_str), title="Logic Timing Diagram"))
