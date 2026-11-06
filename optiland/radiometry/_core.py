"""Shared radiometric array operations and validation."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

from scipy.constants import c, h

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor

if TYPE_CHECKING:
    from optiland._types import BEArray, BEArrayT


def _real_array(values: BEArrayT, name: str, *, active: bool = True) -> None:
    if not isinstance(values, be.ndarray):
        raise TypeError(f"{name} must be a NumPy array or PyTorch tensor.")
    tensor = is_torch_tensor(values)
    if active and tensor != (be.get_backend() == "torch"):
        raise TypeError(f"{name} must belong to the active backend.")
    floating = values.is_floating_point() if tensor else values.dtype.kind == "f"
    if not floating:
        raise TypeError(f"{name} must contain real floating-point values.")


def _validate_image(image: BEArrayT, name: str) -> None:
    _real_array(image, name)
    if image.ndim != 2 or any(size == 0 for size in image.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not bool(be.all(be.isfinite(image))):
        raise ValueError(f"{name} must contain only finite values.")
    if bool(be.any(image < 0)):
        raise ValueError(f"{name} must be nonnegative.")


def _real_scalar(value: float, name: str, *, positive: bool) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}.")
    return value


def _grid_geometry(
    power_like: BEArrayT, *, x_edges_mm: BEArray, y_edges_mm: BEArray
) -> tuple[BEArrayT, BEArrayT, BEArrayT]:
    edges = []
    ny, nx = power_like.shape
    for values, size, name in (
        (x_edges_mm, nx, "x_edges_mm"),
        (y_edges_mm, ny, "y_edges_mm"),
    ):
        _real_array(values, name, active=False)
        if values.ndim != 1 or values.shape[0] != size + 1:
            raise ValueError(f"{name} must have one more edge than image cells.")
        if is_torch_tensor(values):
            if not is_torch_tensor(power_like):
                raise TypeError(f"{name} must belong to the active backend.")
            aligned = values.to(dtype=power_like.dtype, device=power_like.device)
        else:
            aligned = be.asarray(values, dtype=power_like.dtype)
            if is_torch_tensor(power_like):
                aligned = aligned.to(device=power_like.device)
        widths = aligned[1:] - aligned[:-1]
        if not bool(be.all(be.isfinite(aligned))) or not bool(
            be.all(be.isfinite(widths) & (widths > 0))
        ):
            raise ValueError(f"{name} must have finite, strictly increasing edges.")
        edges.append(aligned)
    x, y = edges
    areas = (y[1:] - y[:-1])[:, None] * (x[1:] - x[:-1])[None, :]
    if not bool(be.all(be.isfinite(areas) & (areas > 0))):
        raise ValueError("cell areas must be finite and positive.")
    return x, y, areas


def _calibrated_intensity(intensity: BEArrayT, scale_w_per_mm2: float) -> BEArrayT:
    _validate_image(intensity, "intensity")
    scale = _real_scalar(scale_w_per_mm2, "irradiance_scale_w_per_mm2", positive=False)
    result = intensity * scale
    _validate_image(result, "irradiance")
    return result


def _photon_rate(
    values: BEArrayT, wavelength_m: float, *, watts_per_value: float = 1.0
) -> BEArrayT:
    _validate_image(values, "power")
    wavelength_m = _real_scalar(wavelength_m, "wavelength_m", positive=True)
    scale = _real_scalar(watts_per_value, "watts_per_value", positive=False)
    result = values * (scale * (wavelength_m / (h * c)))
    _validate_image(result, "photon rate")
    return result


def _sum_blocks(values: BEArrayT, factors: tuple[int, int]) -> BEArrayT:
    if not isinstance(factors, tuple) or len(factors) != 2:
        raise TypeError("binning must be a (by, bx) tuple of positive integers.")
    if any(not isinstance(v, Integral) or isinstance(v, bool) for v in factors):
        raise TypeError("binning must contain positive integers.")
    by, bx = (int(v) for v in factors)
    if by <= 0 or bx <= 0:
        raise ValueError("binning factors must be positive.")
    ny, nx = values.shape
    if ny % by or nx % bx:
        raise ValueError("image dimensions must be divisible by binning factors.")
    blocks = be.reshape(values, (ny // by, by, nx // bx, bx))
    result = be.sum(be.sum(blocks, axis=3), axis=1)
    _validate_image(result, "binned image")
    return result
