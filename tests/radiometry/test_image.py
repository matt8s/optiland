from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from tests.utils import assert_allclose, assert_array_equal


def _image(power, x=(0.0, 1.0), y=(0.0, 1.0), wavelength=0.5):
    from optiland.radiometry import RadiometricImage

    return RadiometricImage(
        power,
        x_edges_mm=x if isinstance(x, be.ndarray) else np.asarray(x, dtype=float),
        y_edges_mm=y if isinstance(y, be.ndarray) else np.asarray(y, dtype=float),
        wavelength_um=wavelength,
    )


@pytest.mark.parametrize("precision,rtol", [("float64", 1e-12), ("float32", 5e-7)])
def test_reference_power_area_photons_and_exposure(set_test_backend, precision, rtol):
    try:
        be.set_precision(precision)
        power = be.array([[1e-9]])
        image = _image(power, x=(0, 0.01), y=(0, 0.02))
        assert_allclose(image.irradiance_w_per_mm2, [[5e-6]], rtol=rtol, atol=0)
        assert_allclose(image.total_power_w, 1e-9, rtol=rtol, atol=0)
        assert_allclose(
            image.photon_rate_per_s, [[2.5170582837713547e9]], rtol=rtol, atol=0
        )
        assert_allclose(image.energy_j(0.01), [[1e-11]], rtol=rtol, atol=0)
        assert_allclose(
            image.photon_count(0.01), [[2.5170582837713547e7]], rtol=rtol, atol=0
        )
        assert_array_equal(image.photon_count(0), np.zeros((1, 1)))
        assert image.photon_rate_per_s.dtype == power.dtype
        assert image.energy_j(0.01).dtype == power.dtype
        if be.get_backend() == "torch":
            assert image.photon_rate_per_s.device == power.device
        doubled = _image(power, x=(0, 0.01), y=(0, 0.02), wavelength=1)
        assert_allclose(
            doubled.photon_rate_per_s, 2 * image.photon_rate_per_s, rtol=rtol, atol=0
        )
    finally:
        be.set_precision("float64")


def test_shared_calibration_and_photon_factor(set_test_backend):
    from optiland.radiometry._core import _calibrated_intensity, _photon_rate

    density = _calibrated_intensity(be.ones((2, 3)), 1e-9 / (6 * 0.01 * 0.02))
    rate = _photon_rate(density, 500e-9, watts_per_value=0.01 * 0.02)
    assert_allclose(be.sum(rate), 2.5170582837713547e9, rtol=1e-12, atol=0)
    assert_array_equal(_calibrated_intensity(density, 0), np.zeros((2, 3)))
    for bad in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            _calibrated_intensity(density, bad)
    with pytest.raises(ValueError):
        _calibrated_intensity(be.full((1, 1), np.inf), 0)


def test_unequal_cell_areas_and_block_sums(set_test_backend):
    power = be.array(np.arange(1, 25, dtype=float).reshape(4, 6))
    before = be.to_numpy(power).copy()
    image = _image(power, x=(0, 1, 3, 6, 10, 15, 21), y=(0, 1, 3, 6, 10))
    binned = image.binning((2, 3))
    assert_array_equal(binned.power_w, np.array([[30, 48], [102, 120]]))
    assert_array_equal(binned.x_edges_mm, [0, 6, 21])
    assert_array_equal(binned.y_edges_mm, [0, 3, 10])
    assert_allclose(
        binned.irradiance_w_per_mm2,
        np.array([[30 / 18, 48 / 45], [102 / 42, 120 / 105]]),
        rtol=1e-12,
        atol=0,
    )
    assert_allclose(binned.total_power_w, 300, rtol=0, atol=0)
    assert_allclose(
        be.sum(binned.photon_rate_per_s),
        be.sum(image.photon_rate_per_s),
        rtol=1e-12,
        atol=0,
    )
    assert binned.wavelength_um == image.wavelength_um
    assert_array_equal(power, before)


def test_alias_updates_and_noncontiguous_binning(set_test_backend):
    source = be.array(np.arange(1, 25, dtype=float).reshape(4, 6).T.copy())
    power = be.transpose(source)
    image = _image(power, x=range(7), y=range(5))
    assert_array_equal(image.binning((2, 3)).power_w, np.array([[30, 48], [102, 120]]))
    if be.get_backend() == "torch":
        import torch

        with torch.no_grad():
            power[0, 0] += 1
    else:
        power[0, 0] += 1
    assert_allclose(image.total_power_w, 301, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, -np.inf])
def test_invalid_power(set_test_backend, bad):
    with pytest.raises(ValueError):
        _image(be.array([[bad]]))


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (2,), (1, 1, 1)])
def test_invalid_image_shape(set_test_backend, shape):
    with pytest.raises(ValueError):
        _image(be.ones(shape))


@pytest.mark.parametrize("dtype", ["bool", "int64", "complex128"])
def test_invalid_power_dtype(set_test_backend, dtype):
    power = be.asarray(np.ones((1, 1), dtype=dtype), dtype=getattr(np, dtype))
    with pytest.raises(TypeError):
        _image(power)


@pytest.mark.parametrize("bad", [[[1.0]], 1.0, "image"])
def test_invalid_power_type(set_test_backend, bad):
    with pytest.raises(TypeError):
        _image(bad)


@pytest.mark.parametrize(
    "edges", [(0,), (0, 0), (1, 0), (0, np.inf), (0, np.nan), (0, 1, 2), ((0, 1),)]
)
def test_invalid_edges(set_test_backend, edges):
    with pytest.raises(ValueError):
        _image(be.ones((1, 1)), x=edges)


@pytest.mark.parametrize("value", [True, "1", 1j, [1]])
def test_invalid_scalar_types(set_test_backend, value):
    with pytest.raises(TypeError):
        _image(be.ones((1, 1)), wavelength=value)
    image = _image(be.ones((1, 1)))
    for method in (image.energy_j, image.photon_count):
        with pytest.raises(TypeError):
            method(value)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_invalid_scalar_values(set_test_backend, value):
    with pytest.raises(ValueError):
        _image(be.ones((1, 1)), wavelength=value)
    image = _image(be.ones((1, 1)))
    for method in (image.energy_j, image.photon_count):
        with pytest.raises(ValueError):
            method(value)
    with pytest.raises(ValueError):
        _image(be.ones((1, 1)), wavelength=0)


@pytest.mark.parametrize("factors", [1, [1, 1], (1,), (1, 1, 1), (True, 1), (1.0, 1)])
def test_invalid_binning_types(set_test_backend, factors):
    with pytest.raises(TypeError):
        _image(be.ones((1, 1))).binning(factors)


@pytest.mark.parametrize("factors", [(0, 1), (-1, 1), (2, 1)])
def test_invalid_binning_values(set_test_backend, factors):
    with pytest.raises(ValueError):
        _image(be.ones((1, 1))).binning(factors)


def test_invalid_derived_geometry_and_overflow(set_test_backend):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for x, y in [
            ((-1e308, 1e308), (0, 1)),
            ((0, 1e200), (0, 1e200)),
            ((0, 1e-200), (0, 1e-200)),
        ]:
            with pytest.raises(ValueError):
                _image(be.ones((1, 1)), x=x, y=y)
        image = _image(be.full((1, 2), 1e308), x=(0, 1, 2))
        with pytest.raises(ValueError):
            _ = image.total_power_w
        with pytest.raises(ValueError):
            _ = image.photon_rate_per_s
        with pytest.raises(ValueError):
            image.energy_j(10)
        with pytest.raises(ValueError):
            image.photon_count(10)
        tiny = _image(be.ones((1, 1)), x=(0, 1e-160), y=(0, 1e-160))
        with pytest.raises(ValueError):
            _ = tiny.irradiance_w_per_mm2


def test_mutated_invalid_state_rejected_even_at_zero_exposure(set_test_backend):
    edges = np.array([0.0, 1.0])
    image = _image(be.ones((1, 1)), x=edges)
    if be.get_backend() == "torch":
        import torch

        with torch.no_grad():
            image.power_w[0, 0] = -1
    else:
        image.power_w[0, 0] = -1
    with pytest.raises(ValueError):
        image.energy_j(0)
    image = _image(be.ones((1, 1)))
    if be.get_backend() == "torch":
        with torch.no_grad():
            image.x_edges_mm[1] = 0
    else:
        image.x_edges_mm[1] = 0
    with pytest.raises(ValueError):
        _ = image.irradiance_w_per_mm2


def test_zero_power_is_valid(set_test_backend):
    image = _image(be.zeros((1, 1)))
    assert_array_equal(image.photon_rate_per_s, np.zeros((1, 1)))
    assert_allclose(image.total_power_w, 0, rtol=0, atol=0)


def test_input_precision_and_geometry_graph_survive_default_change(set_test_backend):
    power = be.ones((1, 1))
    edges = np.array([1.0, 1.0 + 1e-9])
    try:
        be.set_precision("float32")
        image = _image(power, x=edges)
        assert image.x_edges_mm.dtype == power.dtype
        assert image.irradiance_w_per_mm2.dtype == power.dtype
        assert_array_equal(image.x_edges_mm, edges)
    finally:
        be.set_precision("float64")
    if be.get_backend() == "torch":
        x = be.array([0.0, 2.0])
        y = be.array([0.0, 3.0])
        image = _image(be.array([[6.0]]), x=x, y=y)
        be.sum(image.irradiance_w_per_mm2).backward()
        assert_allclose(x.grad, [0.5, -0.5], rtol=1e-12, atol=0)
        assert_allclose(y.grad, [1 / 3, -1 / 3], rtol=1e-12, atol=0)
        power = be.ones((1, 1))
        be.sum(_image(power).photon_rate_per_s).backward()
        assert_allclose(power.grad, 2.5170582837713547e18, rtol=1e-12, atol=0)
    for step in (1e-3, 1e-4, 1e-5):
        plus = _image(be.array([[6.0]]), x=(0, 2 + step), y=(0, 3))
        minus = _image(be.array([[6.0]]), x=(0, 2 - step), y=(0, 3))
        derivative = (plus.irradiance_w_per_mm2 - minus.irradiance_w_per_mm2) / (
            2 * step
        )
        assert_allclose(derivative, -0.5, rtol=1e-8 if step == 1e-5 else 1e-5, atol=0)


def test_backend_switch_and_wrong_backend_inputs(set_test_backend):
    if "torch" not in be.list_available_backends():
        pytest.skip("Requires both backends")
    active = be.get_backend()
    power = be.ones((1, 1))
    image = _image(power)
    try:
        be.set_backend("torch" if active == "numpy" else "numpy")
        with pytest.raises(RuntimeError, match="backend"):
            _ = image.total_power_w
        with pytest.raises(TypeError, match="backend"):
            _image(power)
    finally:
        be.set_backend(active)
