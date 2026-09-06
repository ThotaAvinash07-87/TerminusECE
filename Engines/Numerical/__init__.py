"""Numerical computation engine for TerminusECE (MATLAB-like matrix math and transforms)."""

from .parser import NumericalASTParser, NumericalWorkspace
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
from .transforms import (
    TransferFunction,
    DiscreteTransferFunction,
    fourier_transform,
    inverse_fourier_transform,
)

__all__ = [
    "NumericalASTParser",
    "NumericalWorkspace",
    "matrix_det",
    "matrix_inv",
    "matrix_rank",
    "matrix_eig",
    "matrix_svd",
    "matrix_solve",
    "matrix_lu",
    "matrix_qr",
    "poly_roots",
    "poly_conv",
    "TransferFunction",
    "DiscreteTransferFunction",
    "fourier_transform",
    "inverse_fourier_transform",
]
