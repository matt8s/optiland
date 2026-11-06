from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.results import IrradianceMap, SpectralResult
from tests.utils import assert_allclose, assert_array_equal


def _result():
    power = be.array(np.arange(1, 7, dtype=float))
    return IrradianceMap(
        irradiance=np.arange(1, 7, dtype=float).reshape(2, 3),
        x_coords=np.array([0.5, 1.5, 2.5]),
        y_coords=np.array([0.5, 1.5]),
        total_flux=21.0,
        num_rays_hit=6,
        data=power,
    )


def _convert(result, **kwargs):
    from optiland.radiometry import from_nsq_irradiance

    return from_nsq_irradiance(
        result,
        **dict(
            x_edges_mm=np.arange(4, dtype=float),
            y_edges_mm=np.arange(3, dtype=float),
            wavelength_um=0.5,
        )
        | kwargs,
    )


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_attached_buffer_and_gradient(set_test_backend, precision):
    try:
        be.set_precision(precision)
        result = _result()
        before = result.irradiance.copy()
        image = _convert(result)
        assert image.power_w.shape == (2, 3)
        assert image.power_w.dtype == result.data.dtype
        assert_array_equal(image.power_w, before)
        assert_allclose(image.total_power_w, 21, rtol=0, atol=0)
        if be.get_backend() == "torch":
            image.total_power_w.backward()
            assert_array_equal(result.data.grad, np.ones(6))
        assert_array_equal(result.irradiance, before)
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize(
    "member", ["x_coords", "y_coords", "irradiance", "total_flux", "data"]
)
def test_inconsistent_result_rejected(set_test_backend, member):
    result = _result()
    if member == "data":
        result.data = result.data[:-1]
    else:
        setattr(result, member, getattr(result, member) * 1.01)
    with pytest.raises(ValueError):
        _convert(result)


def test_legacy_is_explicitly_forward_only(set_test_backend):
    result = _result()
    result.data = None
    if be.get_backend() == "torch":
        with pytest.raises(ValueError, match="NumPy|numpy"):
            _convert(result)
    else:
        assert_array_equal(_convert(result).power_w, result.irradiance)
        result.total_flux = 42
        with pytest.raises(ValueError):
            _convert(result)


def test_dark_single_pixel_requires_explicit_geometry(set_test_backend):
    from optiland.radiometry import from_nsq_irradiance

    result = IrradianceMap(
        np.zeros((1, 1)), np.zeros(1), np.zeros(1), 0, 0, be.zeros(1)
    )
    x, y = np.array([-0.1, 0.1]), np.array([-0.2, 0.2])
    kwargs = dict(x_edges_mm=x, y_edges_mm=y, wavelength_um=0.5)
    image = from_nsq_irradiance(result, **kwargs)
    assert_array_equal(image.photon_rate_per_s, np.zeros((1, 1)))
    assert_array_equal(image.x_edges_mm, x)
    assert_array_equal(image.y_edges_mm, y)
    for name in kwargs:
        with pytest.raises(TypeError):
            from_nsq_irradiance(
                result, **{k: v for k, v in kwargs.items() if k != name}
            )
    result.irradiance = np.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        from_nsq_irradiance(result, **kwargs)
    spectral = SpectralResult(
        np.ones((1, 1, 2)), np.zeros(1), np.zeros(1), np.array([0.5, 0.6]), 2, 2
    )
    with pytest.raises(TypeError):
        from_nsq_irradiance(spectral, **kwargs)


def test_native_detector_power_and_photons(set_test_backend):
    from optiland.radiometry import from_nsq_irradiance

    flux = be.array(1e-9)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.5), total_flux=flux, aperture_radius=1
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=10),
        IrradianceDetectorConfig(
            width=10, height=8, num_pixels_x=12, num_pixels_y=8, splat="bilinear"
        ),
    )
    backend = (
        NumpyBackend(seed=0) if be.get_backend() == "numpy" else TorchBackend(seed=0)
    )
    result = scene.trace(num_rays=512, seed=0, backend=backend).detectors["D"]
    image = from_nsq_irradiance(
        result,
        x_edges_mm=np.linspace(-5, 5, 13),
        y_edges_mm=np.linspace(-4, 4, 9),
        wavelength_um=0.5,
    )
    assert_allclose(image.total_power_w, 1e-9, rtol=1e-12, atol=0)
    assert_allclose(
        be.sum(image.photon_rate_per_s), 2.5170582837713547e9, rtol=1e-12, atol=0
    )
    if be.get_backend() == "torch":
        import torch

        power_grad = torch.autograd.grad(image.total_power_w, flux, retain_graph=True)[
            0
        ]
        photon_grad = torch.autograd.grad(be.sum(image.photon_rate_per_s), flux)[0]
        assert_allclose(power_grad, 1, rtol=1e-12, atol=0)
        assert_allclose(photon_grad, 2.5170582837713547e18, rtol=1e-12, atol=0)


@pytest.mark.parametrize("precision,rtol", [("float64", 1e-12), ("float32", 5e-7)])
def test_native_high_resolution_roundoff_and_stale_snapshot_rejection(
    set_test_backend, precision, rtol
):
    from optiland.radiometry import from_nsq_irradiance

    try:
        be.set_precision(precision)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.5),
                total_flux=1e-9,
                aperture_radius=2,
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=10),
            IrradianceDetectorConfig(
                width=6.4,
                height=4.8,
                num_pixels_x=128,
                num_pixels_y=128,
                splat="bilinear",
            ),
        )
        backend = (
            NumpyBackend(seed=0)
            if be.get_backend() == "numpy"
            else TorchBackend(seed=0)
        )
        result = scene.trace(num_rays=512, seed=0, backend=backend).detectors["D"]
        kwargs = dict(
            x_edges_mm=np.linspace(-3.2, 3.2, 129),
            y_edges_mm=np.linspace(-2.4, 2.4, 129),
            wavelength_um=0.5,
        )
        image = from_nsq_irradiance(result, **kwargs)
        assert_allclose(image.total_power_w, 1e-9, rtol=rtol, atol=0)
        assert_allclose(
            be.sum(image.photon_rate_per_s), 2.5170582837713547e9, rtol=rtol, atol=0
        )
        result.irradiance *= 1.01
        with pytest.raises(ValueError, match="irradiance"):
            from_nsq_irradiance(result, **kwargs)
    finally:
        be.set_precision("float64")
