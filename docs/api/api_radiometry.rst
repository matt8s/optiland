.. _api_radiometry:

Radiometry
==========

``RadiometricImage`` expresses a monochromatic planar measurement as power in
each rectangular cell, with explicit cell edges in mm and vacuum wavelength in
µm. It provides irradiance in W/mm², total captured power in W, photon rates in
photons/s, and exposure-integrated energy in J and expected photon counts.
See the :doc:`native examples <../gallery/analysis/radiometry>` for ray and
scalar-field calculations using the same interface.

Calibration and geometry
------------------------

Arrays use ``(ny, nx)`` order. Cell widths may differ; irradiance is cell power
divided by its area. ``binning((by, bx))`` sums powers in aligned blocks and
retains the outer edges, so total power and physical registration are preserved.
Both dimensions must divide exactly by their positive integer factors.

The adapters retain the producers' physical meaning:

* ``from_nsq_irradiance`` uses an ``IrradianceMap``'s attached pixel-power buffer
  and checks its coordinates, irradiance, and total against supplied detector
  edges. A legacy result without that buffer is supported only on NumPy and is
  forward-only. Its geometry cannot be inferred from a dark or single-cell map.
* ``from_sequential_irradiance`` accepts one selected analysis entry, transposes
  its ``(nx, ny)`` density, and applies the declared watts per ray-weight unit.
  Already watt-calibrated scalar rays use a factor of one. Ordinary unit-weight
  pupil rays need a launch calibration; polarized field weights require a
  separate physical interpretation.
* ``from_scalar_field`` applies a fixed W/mm² calibration to squared amplitude.
  ``length_unit_mm`` converts both sampling and vacuum wavelength from the
  field's length unit. Samples approximate cell-mean irradiance by rectangle
  quadrature, so refine the grid to check convergence. Perform coherent
  superposition and propagation before conversion. The calibration must be
  appropriate to the physical regime, such as a paraxial beam on a transverse
  plane; a general electromagnetic normal-flux factor is not assumed.

Calibration stays fixed through attenuation and changes in captured area.
No adapter renormalizes surviving samples to the launched power. A declared
wavelength applies to the entire image: the caller must know the result is
monochromatic, because a two-dimensional irradiance map cannot reveal a hidden
spectral mixture. Zero power and zero exposure are valid. Photon counts are
expectations for stationary illumination, before quantum efficiency, shot noise,
or electronic response.

Arrays and gradients
--------------------

Power and edge arrays may share input storage. Derived quantities are validated
and recomputed on access; they are not cached snapshots. Do not modify tensors
needed by an outstanding autograd calculation. Numerical outputs preserve the
input floating precision, device, and available gradient connections, including
connected geometry. Validation may synchronize device execution. The adapter
cannot restore gradients lost by a producer or create visibility gradients that
a tracer does not provide. Changing the active backend after construction raises
an error when a derived measurement is requested.

Spectral and photometric results
-------------------------------

This interface accepts two-dimensional monochromatic images. Existing NSQ
``SpectralResult`` irradiance values are already integrated within each
wavelength bin; multiplying them by a bin width again would double-count it.
A future spectral adapter must retain bin edges and representation. Exact
photon conversion needs the wavelength-weighted power integral; bin-center
conversion is an approximation. Existing CIE-weighted photometry is available
through ``optiland.nonsequential.units.to_photometric``.

Reference
---------

.. autoclass:: optiland.radiometry.RadiometricImage
   :members:

.. autofunction:: optiland.radiometry.from_nsq_irradiance

.. autofunction:: optiland.radiometry.from_sequential_irradiance

.. autofunction:: optiland.radiometry.from_scalar_field
