"""Scalar physical-optics field models and propagation algorithms."""

from __future__ import annotations

from .diagnostics import BoundaryDiagnostic, boundary_diagnostic
from .field import ScalarField, gaussian_field
from .propagation import angular_spectrum
from .train import ScalarOpticalTrain

__all__ = [
    "BoundaryDiagnostic",
    "ScalarField",
    "ScalarOpticalTrain",
    "angular_spectrum",
    "boundary_diagnostic",
    "gaussian_field",
]
