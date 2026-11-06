from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.nonsequential.results import IrradianceMap
from optiland.physical_optics import ScalarField
from optiland.radiometry import (
    from_nsq_irradiance,
    from_scalar_field,
    from_sequential_irradiance,
)
from tests.utils import assert_allclose


def test_native_producers_agree_with_si_reference(set_test_backend):
    x = np.array([-0.015, -0.005, 0.005, 0.015])
    y = np.array([-0.02, 0.0, 0.02])
    density = 1e-9 / (6 * 0.01 * 0.02)
    scalar = from_scalar_field(
        ScalarField(be.ones((2, 3)), dx=0.01, dy=0.02, wavelength=0.0005),
        length_unit_mm=1,
        irradiance_scale_w_per_mm2=density,
    )
    sequential = from_sequential_irradiance(
        be.full((3, 2), density),
        x_edges_mm=x,
        y_edges_mm=y,
        wavelength_um=0.5,
        watts_per_ray_weight=1,
    )
    nsq = from_nsq_irradiance(
        IrradianceMap(
            np.full((2, 3), density),
            np.array([-0.01, 0, 0.01]),
            np.array([-0.01, 0.01]),
            1e-9,
            6,
            be.full((6,), 1e-9 / 6),
        ),
        x_edges_mm=x,
        y_edges_mm=y,
        wavelength_um=0.5,
    )
    for image in (scalar, sequential, nsq):
        assert_allclose(image.total_power_w, 1e-9, rtol=1e-12, atol=0)
        assert_allclose(
            be.sum(image.photon_rate_per_s), 2.5170582837713547e9, rtol=1e-12, atol=0
        )
        assert_allclose(image.power_w, np.full((2, 3), 1e-9 / 6), rtol=1e-12, atol=0)
        assert_allclose(image.power_w, scalar.power_w, rtol=1e-12, atol=0)
        binned = image.binning((2, 3))
        assert_allclose(binned.power_w, np.array([[1e-9]]), rtol=1e-12, atol=0)
        assert_allclose(binned.energy_j(0.01), np.array([[1e-11]]), rtol=1e-12, atol=0)
        assert_allclose(
            binned.photon_count(0.01),
            np.array([[2.5170582837713547e7]]),
            rtol=1e-12,
            atol=0,
        )


def test_native_gallery_code_runs(set_test_backend):
    path = (
        Path(__file__).resolve().parents[2] / "docs/gallery/analysis/radiometry.ipynb"
    )
    notebook = json.loads(path.read_text())
    namespace = {}
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
        assert_allclose(namespace["nsq_image"].total_power_w, 1e-9, rtol=1e-12, atol=0)
        assert (
            0 < float(be.to_numpy(namespace["sequential_image"].total_power_w)) <= 1e-9
        )
        assert_allclose(
            namespace["scalar_image"].total_power_w,
            1.5197119170278266e-11,
            rtol=2.1e-4,
            atol=0,
        )
    finally:
        plt.close("all")
