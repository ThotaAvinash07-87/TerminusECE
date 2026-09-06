"""CORE utilities and common infrastructure for TerminusECE."""

from .common_math import parse_eng_unit, format_eng_unit, Waveform, SignalMetrics, split_smart_statements
from .ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter, RoutePath
from .ipc_router import IPCRouter, IPCClient

__all__ = [
    "parse_eng_unit",
    "format_eng_unit",
    "Waveform",
    "SignalMetrics",
    "split_smart_statements",
    "AsciiCanvas",
    "AsciiPlotter",
    "AsciiBodePlotter",
    "RoutePath",
    "IPCRouter",
    "IPCClient",
]
