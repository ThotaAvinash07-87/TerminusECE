"""AST Parser and interactive Workspace for MATLAB-style numerical expressions."""

from __future__ import annotations
import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from CORE.common_math import Waveform, parse_eng_unit
from .matrix_ops import (
    matrix_det,
    matrix_inv,
    matrix_rank,
    matrix_eig,
    matrix_svd,
    matrix_solve,
    matrix_lu,
    matrix_qr,
    poly_roots,
    poly_conv,
)
from .transforms import TransferFunction, DiscreteTransferFunction, fourier_transform


class NumericalWorkspace:
    """Variable memory store and computation context for the Numerical engine."""

    def __init__(self):
        self.variables: Dict[str, Any] = {
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
        }
        self._init_builtins()

    def _init_builtins(self) -> None:
        self.builtins: Dict[str, Any] = {
            # Constants
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
            # Math
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "angle": lambda x: np.angle(x, deg=True),
            "real": np.real,
            "imag": np.imag,
            "conj": np.conj,
            # Array creation
            "linspace": np.linspace,
            "arange": np.arange,
            "zeros": lambda *args: np.zeros(args if len(args) > 1 else (args[0], args[0])),
            "ones": lambda *args: np.ones(args if len(args) > 1 else (args[0], args[0])),
            "eye": np.eye,
            # Matrix operations
            "det": matrix_det,
            "inv": matrix_inv,
            "rank": matrix_rank,
            "eig": matrix_eig,
            "svd": matrix_svd,
            "solve": matrix_solve,
            "lu": matrix_lu,
            "qr": matrix_qr,
            "roots": poly_roots,
            "conv": poly_conv,
            # Statistics
            "sum": np.sum,
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            # Transforms & Control
            "tf": lambda num, den, name="H(s)": TransferFunction(num, den, name=name),
            "tf_d": lambda b, a, dt=1.0, name="H(z)": DiscreteTransferFunction(b, a, dt=dt, name=name),
            "fft": lambda wf: wf.compute_fft() if isinstance(wf, Waveform) else np.fft.rfft(wf),
        }

    def get_eval_context(self) -> Dict[str, Any]:
        ctx = dict(self.builtins)
        ctx.update(self.variables)
        return ctx

    def clear(self) -> None:
        self.variables = {
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
        }

    def list_vars(self) -> Dict[str, str]:
        res = {}
        for k, v in self.variables.items():
            if k in ("pi", "e", "j", "i"):
                continue
            if isinstance(v, np.ndarray):
                res[k] = f"Array {v.shape} {v.dtype}"
            elif isinstance(v, (TransferFunction, DiscreteTransferFunction)):
                res[k] = repr(v)
            elif isinstance(v, Waveform):
                res[k] = f"Waveform ({len(v.x)} pts)"
            else:
                res[k] = f"{type(v).__name__}: {v}"
        return res


class NumericalASTParser:
    """Evaluates mathematical strings, matrix syntax, assignments, and expressions."""

    def __init__(self, workspace: Optional[NumericalWorkspace] = None):
        self.workspace = workspace or NumericalWorkspace()

    def _preprocess_syntax(self, code: str) -> str:
        """Translates MATLAB-style matrix syntax like `[1 2; 3 4]` or `1:0.5:10` to valid Python."""
        text = code.strip()

        # Handle colon range: start:step:stop or start:stop
        def colon_repl(match: re.Match) -> str:
            parts = match.group(0).split(":")
            if len(parts) == 2:
                return f"arange({parts[0]}, {parts[1]} + 1e-9, 1)"
            elif len(parts) == 3:
                return f"arange({parts[0]}, {parts[2]} + 1e-9, {parts[1]})"
            return match.group(0)

        text = re.sub(r'(?<![a-zA-Z0-9_])\b\d+(?:\.\d+)?:\d+(?:\.\d+)?(?::\d+(?:\.\d+)?)?\b', colon_repl, text)

        # Handle matrix brackets: e.g. [1 2; 3 4] -> np.array([[1, 2], [3, 4]])
        def matrix_bracket_repl(match: re.Match) -> str:
            inner = match.group(1).strip()
            if not inner:
                return "np.array([])"
            rows = inner.split(";")
            row_lists = []
            for r in rows:
                r_clean = r.strip()
                # Split by space or comma
                elements = [e.strip() for e in re.split(r'[\s,]+', r_clean) if e.strip()]
                row_lists.append(f"[{', '.join(elements)}]")
            if len(row_lists) == 1 and not inner.endswith(";"):
                return f"np.array({row_lists[0]})"
            return f"np.array([{', '.join(row_lists)}])"

        text = re.sub(r'\[([^\]]+)\]', matrix_bracket_repl, text)

        # Element-wise power .^ -> **
        text = text.replace(".^", "**")
        # Element-wise multiply .* -> *
        text = text.replace(".*", "*")
        # Element-wise divide ./ -> /
        text = text.replace("./", "/")
        # MATLAB power ^ -> **
        text = re.sub(r'(?<=[a-zA-Z0-9_\)\]])\^(?=[a-zA-Z0-9_\(\[])', '**', text)

        return text

    def execute(self, code: str) -> Any:
        """Executes a MATLAB-like line or block, assigning variables or returning results."""
        line = code.strip()
        if not line:
            return None

        # Check special commands
        if line.lower() in ("clear", "clear all"):
            self.workspace.clear()
            return "Workspace cleared."
        if line.lower() in ("who", "whos"):
            return self.workspace.list_vars()

        # Variable assignment detection: var_name = expr
        assign_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if assign_match:
            var_name, expr_str = assign_match.groups()
            py_code = self._preprocess_syntax(expr_str)
            ctx = self.workspace.get_eval_context()
            ctx["np"] = np
            val = eval(py_code, {"__builtins__": {}}, ctx)
            self.workspace.variables[var_name] = val
            return val

        # Pure expression evaluation
        py_code = self._preprocess_syntax(line)
        ctx = self.workspace.get_eval_context()
        ctx["np"] = np
        return eval(py_code, {"__builtins__": {}}, ctx)
