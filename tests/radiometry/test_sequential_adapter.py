from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis import IncoherentIrradiance
from optiland.sources import SMFSource
from tests.test_extended_sources import _simple_singlet_with_detector
from tests.utils import assert_allclose, assert_array_equal


def _convert(density, **kwargs):
    from optiland.radiometry import from_sequential_irradiance

    return from_sequential_irradiance(
        density,
        **dict(
            x_edges_mm=np.array([0.0, 1.0, 3.0, 6.0]),
            y_edges_mm=np.array([0.0, 2.0, 5.0]),
            wavelength_um=0.5,
            watts_per_ray_weight=0.5,
        )
        | kwargs,
    )


def test_asymmetric_axes_areas_and_calibration(set_test_backend):
    density = be.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    before = be.to_numpy(density).copy()
    image = _convert(density)
    assert_array_equal(image.power_w, np.array([[1, 4, 9], [6, 15, 27]]))
    assert_allclose(image.total_power_w, 62, rtol=0, atol=0)
    assert_array_equal(
        image.irradiance_w_per_mm2,
        np.array([[0.5, 1, 1.5], [2, 2.5, 3]]),
    )
    assert_allclose(_convert(density * 0.25).total_power_w, 15.5, rtol=0, atol=0)
    assert_array_equal(
        _convert(density, watts_per_ray_weight=0).power_w, np.zeros((2, 3))
    )
    if be.get_backend() == "torch":
        image.total_power_w.backward()
        assert_array_equal(density.grad, np.array([[1, 1.5], [2, 3], [3, 4.5]]))
    assert_array_equal(density, before)


@pytest.mark.parametrize("scale", [-1, np.nan, np.inf])
def test_invalid_calibration(set_test_backend, scale):
    with pytest.raises(ValueError):
        _convert(be.ones((3, 2)), watts_per_ray_weight=scale)


def test_calibration_required_and_invalid_density_not_hidden(set_test_backend):
    from optiland.radiometry import from_sequential_irradiance

    with pytest.raises(TypeError):
        from_sequential_irradiance(
            be.ones((3, 2)),
            x_edges_mm=np.arange(4, dtype=float),
            y_edges_mm=np.arange(3, dtype=float),
            wavelength_um=0.5,
        )
    with pytest.raises(TypeError):
        _convert(be.ones((3, 2)), watts_per_ray_weight=True)
    for value in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            _convert(be.full((3, 2), value), watts_per_ray_weight=0)


def test_native_source_captured_power(set_test_backend):
    from optiland.radiometry import from_sequential_irradiance

    optic = _simple_singlet_with_detector()
    source = SMFSource(mfd_um=10.4, wavelength_um=0.55, total_power=1e-9)
    analysis = IncoherentIrradiance(optic, source=source, num_rays=128, res=(12, 8))
    density, x_edges, y_edges = analysis.data[0][0]
    image = from_sequential_irradiance(
        density,
        x_edges_mm=x_edges,
        y_edges_mm=y_edges,
        wavelength_um=0.55,
        watts_per_ray_weight=1,
    )
    # This fixture's untilted image plane has the same local and global x/y.
    detector = optic.surfaces[-1]
    inside = (
        (detector.x >= x_edges[0])
        & (detector.x <= x_edges[-1])
        & (detector.y >= y_edges[0])
        & (detector.y <= y_edges[-1])
        & (detector.intensity > 0)
    )
    captured = be.sum(detector.intensity[inside])
    assert image.power_w.shape == (8, 12)
    assert 0 < float(be.to_numpy(captured)) <= 1e-9
    assert_allclose(image.total_power_w, captured, rtol=1e-12, atol=0)
    assert_allclose(
        be.sum(image.photon_rate_per_s),
        captured * 2.7687641121484902e18,
        rtol=1e-12,
        atol=0,
    )
