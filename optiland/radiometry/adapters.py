"""Explicit calibration and geometry adapters for existing optical results."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor
from optiland.radiometry._core import (
    _grid_geometry,
    _real_array,
    _real_scalar,
    _validate_image,
)
from optiland.radiometry.image import RadiometricImage
from optiland.utils import machine_eps

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArrayT
    from optiland.nonsequential.results import IrradianceMap


def _consistent(
    actual: ScalarOrArrayT,
    reference: ScalarOrArrayT,
    name: str,
    *,
    rtol: float = 0,
    atol: ScalarOrArrayT = 0,
) -> None:
    if isinstance(reference, be.ndarray):
        _real_array(reference, name, active=False)
    else:
        _real_scalar(reference, name, positive=False)
    expected = be.asarray(reference, dtype=actual.dtype)
    if is_torch_tensor(actual):
        expected = expected.to(device=actual.device)
    if actual.shape != expected.shape or not bool(
        be.all(
            be.isfinite(expected)
            & (be.abs(actual - expected) <= atol + rtol * be.abs(expected))
        )
    ):
        raise ValueError(f"{name} is inconsistent with pixel power and geometry.")


def from_nsq_irradiance(
    result: IrradianceMap,
    *,
    x_edges_mm: BEArray,
    y_edges_mm: BEArray,
    wavelength_um: float,
) -> RadiometricImage:
    """Adapt a monochromatic non-sequential irradiance result.

    Args:
        result: Existing ``IrradianceMap`` with pixel power in its attached
            ``data`` buffer, or a legacy NumPy-only irradiance result.
        x_edges_mm: Known local detector cell edges along x in mm.
        y_edges_mm: Known local detector cell edges along y in mm.
        wavelength_um: Declared monochromatic vacuum wavelength in micrometers.

    Returns:
        Calibrated measurement retaining the attached power graph when present.

    Notes:
        The result does not retain its spectrum or cell edges. The caller must
        supply the actual geometry and a known single wavelength; this adapter
        cannot detect an undeclared spectral mixture. Coordinates, irradiance,
        and total flux are checked for consistency. Legacy results require the
        NumPy backend because their lost gradients cannot be reconstructed.
    """
    from optiland.nonsequential.results import IrradianceMap

    if not isinstance(result, IrradianceMap):
        raise TypeError("result must be an IrradianceMap.")
    _real_array(result.irradiance, "irradiance", active=False)
    if result.irradiance.ndim != 2 or any(n == 0 for n in result.irradiance.shape):
        raise ValueError("irradiance must be a nonempty two-dimensional array.")
    ny, nx = result.irradiance.shape
    if result.data is None:
        if be.get_backend() != "numpy":
            raise ValueError("Legacy irradiance results require the NumPy backend.")
        _validate_image(result.irradiance, "irradiance")
        _, _, areas = _grid_geometry(
            result.irradiance, x_edges_mm=x_edges_mm, y_edges_mm=y_edges_mm
        )
        power = result.irradiance * areas
    else:
        _real_array(result.data, "data")
        if result.data.ndim != 1 or result.data.shape[0] != ny * nx:
            raise ValueError("data must be the flat pixel-power buffer.")
        power = be.reshape(result.data, (ny, nx))
    image = RadiometricImage(
        power, x_edges_mm=x_edges_mm, y_edges_mm=y_edges_mm, wavelength_um=wavelength_um
    )
    eps = max(
        machine_eps(value)
        for value in (
            power,
            result.irradiance,
            result.x_coords,
            result.y_coords,
            result.total_flux,
        )
    )
    for edges, centers, name in (
        (image.x_edges_mm, result.x_coords, "x_coords"),
        (image.y_edges_mm, result.y_coords, "y_coords"),
    ):
        scale = be.maximum(be.max(be.abs(edges)), edges[-1] - edges[0])
        _consistent((edges[:-1] + edges[1:]) / 2, centers, name, atol=8 * eps * scale)
    _consistent(
        image.irradiance_w_per_mm2, result.irradiance, "irradiance", rtol=16 * eps
    )
    _consistent(
        image.total_power_w,
        result.total_flux,
        "total_flux",
        rtol=64 * eps * max(1, math.ceil(math.log2(nx * ny + 1))),
    )
    return image
