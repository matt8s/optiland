from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField, gaussian_field
from tests.utils import assert_allclose, assert_array_equal


def _convert(field, **kwargs):
    from optiland.radiometry import from_scalar_field

    return from_scalar_field(
        field,
        **dict(length_unit_mm=1, irradiance_scale_w_per_mm2=2) | kwargs,
    )


@pytest.mark.parametrize(
    "unit,dx,dy,wavelength", [(0.001, 2, 3, 0.5), (1, 0.002, 0.003, 0.0005)]
)
def test_physical_units_and_cell_geometry(set_test_backend, unit, dx, dy, wavelength):
    field = ScalarField(be.ones((2, 3)) * (1 + 2j), dx=dx, dy=dy, wavelength=wavelength)
    before = be.to_numpy(field.data).copy()
    image = _convert(field, length_unit_mm=unit, irradiance_scale_w_per_mm2=0.4)
    assert_allclose(
        image.x_edges_mm, [-0.003, -0.001, 0.001, 0.003], rtol=1e-12, atol=0
    )
    assert_allclose(image.y_edges_mm, [-0.003, 0, 0.003], rtol=1e-12, atol=0)
    assert_allclose(image.irradiance_w_per_mm2, 2, rtol=1e-12, atol=0)
    assert_allclose(image.power_w, 1.2e-5, rtol=1e-12, atol=0)
    assert_allclose(image.total_power_w, 7.2e-5, rtol=1e-12, atol=0)
    assert image.wavelength_um == 0.5
    assert_array_equal(field.data, before)


def test_coherence_and_fixed_calibration_through_loss(set_test_backend):
    amplitude = be.ones((4, 6))

    def power(data):
        return _convert(ScalarField(data, dx=1, wavelength=0.0005)).total_power_w

    assert_allclose(power(amplitude), 48, rtol=1e-12, atol=0)
    assert_allclose(power(amplitude * np.exp(0.7j)), 48, rtol=1e-12, atol=0)
    assert_allclose(power(amplitude + amplitude), 192, rtol=1e-12, atol=0)
    assert_allclose(power(amplitude - amplitude), 0, rtol=0, atol=0)
    assert_allclose(power(amplitude * 0.5), 12, rtol=1e-12, atol=0)
    assert_allclose(power(amplitude[:2, :3]), 12, rtol=1e-12, atol=0)
    if be.get_backend() == "torch":
        power(amplitude).backward()
        assert_allclose(amplitude.grad, 4, rtol=1e-12, atol=0)
        common = be.array(1.0)
        power(be.ones((4, 6)) * common).backward()
        assert_allclose(common.grad, 96, rtol=1e-12, atol=0)
    for step in (1e-3, 1e-4, 1e-5):
        derivative = (power(amplitude * (1 + step)) - power(amplitude * (1 - step))) / (
            2 * step
        )
        assert_allclose(derivative, 96, rtol=1e-8, atol=0)


def test_finite_window_gaussian_midpoint_convergence(set_test_backend):
    reference = 1.5197119170278266e-11
    errors = []
    for n in (8, 16, 32):
        field = gaussian_field((n, n), dx=0.24 / n, wavelength=0.0005, waist_radius=0.1)
        image = _convert(field, irradiance_scale_w_per_mm2=1e-9)
        errors.append(abs(float(be.to_numpy(image.total_power_w)) - reference))
    assert 3.8 < errors[0] / errors[1] < 4.1
    assert 3.8 < errors[1] / errors[2] < 4.1
    assert errors[-1] / reference < 2.1e-4


@pytest.mark.parametrize("name", ["length_unit_mm", "irradiance_scale_w_per_mm2"])
@pytest.mark.parametrize(
    "value,error",
    [(-1, ValueError), (np.inf, ValueError), (np.nan, ValueError), (True, TypeError)],
)
def test_invalid_calibration_parameters(set_test_backend, name, value, error):
    field = ScalarField(be.ones((2, 3)), dx=1, wavelength=0.0005)
    with pytest.raises(error):
        _convert(field, **{name: value})


def test_zero_calibration_and_invalid_field(set_test_backend):
    field = ScalarField(be.ones((2, 3)), dx=1, wavelength=0.0005)
    assert_array_equal(
        _convert(field, irradiance_scale_w_per_mm2=0).power_w, np.zeros((2, 3))
    )
    with pytest.raises(ValueError):
        _convert(field, length_unit_mm=0)
    with pytest.raises(TypeError):
        _convert(field.data)
    for value in (np.nan, np.inf):
        field = ScalarField(be.full((2, 3), value), dx=1, wavelength=0.0005)
        with pytest.raises(ValueError):
            _convert(field, irradiance_scale_w_per_mm2=0)


def test_scalar_backend_switch_rejected(set_test_backend):
    if "torch" not in be.list_available_backends():
        pytest.skip("Requires both backends")
    active = be.get_backend()
    field = ScalarField(be.ones((2, 3)), dx=1, wavelength=0.0005)
    try:
        be.set_backend("torch" if active == "numpy" else "numpy")
        with pytest.raises(RuntimeError, match="backend"):
            _convert(field)
    finally:
        be.set_backend(active)
