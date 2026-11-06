"""Calibrated monochromatic measurements on a rectangular plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import optiland.backend as be
from optiland._types import BEArrayT
from optiland.radiometry._core import (
    _grid_geometry,
    _photon_rate,
    _real_scalar,
    _sum_blocks,
    _validate_image,
)

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArrayT


class RadiometricImage(Generic[BEArrayT]):
    """Monochromatic pixel powers with explicit local-plane geometry.

    Args:
        power_w: Nonnegative finite powers in W, shape ``(ny, nx)``.
        x_edges_mm: Strictly increasing x cell edges in mm, length ``nx + 1``.
        y_edges_mm: Strictly increasing y cell edges in mm, length ``ny + 1``.
        wavelength_um: Positive vacuum wavelength in micrometers. Supplying it
            declares that the entire image is monochromatic.

    Notes:
        Power and edge arrays may share input storage. Derived values are
        recomputed and validated on access; callers must not modify tensors
        needed by an outstanding autograd computation. Conversion preserves
        existing gradients but cannot recover gradients lost by a producer.
        Validation can synchronize devices. Geometry is an orthogonal local
        plane; no registration, cropping, or throughput normalization is applied.
    """

    def __init__(
        self,
        power_w: BEArrayT,
        *,
        x_edges_mm: BEArray,
        y_edges_mm: BEArray,
        wavelength_um: float,
    ) -> None:
        """Validate power, sampling, and the declared monochromatic wavelength."""
        _validate_image(power_w, "power_w")
        self._backend = be.get_backend()
        self._power = power_w
        self._wavelength = _real_scalar(wavelength_um, "wavelength_um", positive=True)
        self._x, self._y, _ = _grid_geometry(
            power_w, x_edges_mm=x_edges_mm, y_edges_mm=y_edges_mm
        )

    def _areas(self) -> BEArrayT:
        if be.get_backend() != self._backend:
            raise RuntimeError("the active backend changed after image construction")
        _validate_image(self._power, "power_w")
        return _grid_geometry(self._power, x_edges_mm=self._x, y_edges_mm=self._y)[2]

    @property
    def power_w(self) -> BEArrayT:
        """Return the shared per-cell power array in W, ordered as ``(y, x)``."""
        return self._power

    @property
    def x_edges_mm(self) -> BEArrayT:
        """Return the local x cell edges in mm."""
        return self._x

    @property
    def y_edges_mm(self) -> BEArrayT:
        """Return the local y cell edges in mm."""
        return self._y

    @property
    def wavelength_um(self) -> float:
        """Return the declared vacuum wavelength in micrometers."""
        return self._wavelength

    @property
    def irradiance_w_per_mm2(self) -> BEArrayT:
        """Return each cell's mean irradiance in W/mm²."""
        result = self._power / self._areas()
        _validate_image(result, "irradiance")
        return result

    @property
    def total_power_w(self) -> ScalarOrArrayT:
        """Return total captured power in W, retaining the backend graph."""
        self._areas()
        total = be.sum(self._power)
        if not bool(be.isfinite(total)):
            raise ValueError("total power must be finite.")
        return total

    @property
    def photon_rate_per_s(self) -> BEArrayT:
        """Return expected incident photons per cell per second."""
        self._areas()
        return _photon_rate(self._power, self._wavelength * 1e-6)

    def energy_j(self, exposure_s: float) -> BEArrayT:
        """Integrate stationary power into cell energies in joules.

        Args:
            exposure_s: Finite, nonnegative exposure duration in seconds.

        Returns:
            Energy in J per cell, with the image's shape and backend.
        """
        self._areas()
        duration = _real_scalar(exposure_s, "exposure_s", positive=False)
        result = self._power * duration
        _validate_image(result, "energy")
        return result

    def photon_count(self, exposure_s: float) -> BEArrayT:
        """Integrate stationary photon rate into expected incident counts.

        Args:
            exposure_s: Finite, nonnegative exposure duration in seconds.

        Returns:
            Expected photons per cell, without detector response or shot noise.
        """
        duration = _real_scalar(exposure_s, "exposure_s", positive=False)
        result = self.photon_rate_per_s * duration
        _validate_image(result, "photon count")
        return result

    def binning(self, factors: tuple[int, int]) -> RadiometricImage[BEArrayT]:
        """Sum powers in aligned blocks while retaining their physical edges.

        Args:
            factors: Positive integer ``(by, bx)`` factors dividing both axes.

        Returns:
            A measurement with summed block powers and unchanged wavelength.
        """
        self._areas()
        power = _sum_blocks(self._power, factors)
        by, bx = factors
        return RadiometricImage(
            power,
            x_edges_mm=self._x[::bx],
            y_edges_mm=self._y[::by],
            wavelength_um=self._wavelength,
        )
