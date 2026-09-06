"""Linear algebra and matrix mathematics operations powered by NumPy and SciPy."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import scipy.linalg as la


def matrix_det(a: np.ndarray) -> float:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Determinant requires square matrix, got shape {arr.shape}")
    return float(np.linalg.det(arr))


def matrix_inv(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Inverse requires square matrix, got shape {arr.shape}")
    return np.linalg.inv(arr)


def matrix_rank(a: np.ndarray) -> int:
    arr = np.asarray(a)
    return int(np.linalg.matrix_rank(arr))


def matrix_eig(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(a, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(arr)
    return eigenvalues, eigenvectors


def matrix_svd(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(a, dtype=float)
    u, s, vh = np.linalg.svd(arr)
    return u, s, vh


def matrix_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_mat = np.asarray(a, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    return np.linalg.solve(a_mat, b_vec)


def matrix_lu(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(a, dtype=float)
    p, l, u = la.lu(arr)
    return p, l, u


def matrix_qr(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(a, dtype=float)
    q, r = np.linalg.qr(arr)
    return q, r


def poly_roots(coeffs: Sequence[float]) -> np.ndarray:
    return np.roots(np.asarray(coeffs, dtype=float))


def poly_conv(p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
    return np.convolve(np.asarray(p1, dtype=float), np.asarray(p2, dtype=float))
