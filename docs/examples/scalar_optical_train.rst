Scalar optical train contract
=============================

``ScalarOpticalTrain`` applies native surface sag, apertures, and supported
analytic phase profiles to an arbitrary sampled ``ScalarField``.
It uses a scalar paraxial phase-screen approximation at each vertex and the
existing angular-spectrum propagator between vertices.
The surface approximation does not become exact merely by refining the grid.

For a lossless lens followed by a homogeneous gap::

   import numpy as np
   from optiland.optic import Optic
   from optiland.physical_optics import ScalarOpticalTrain, gaussian_field

   optic = Optic()
   optic.surfaces.add(index=0, thickness=np.inf)
   optic.surfaces.add(index=1, z=0, surface_type="paraxial", f=50)
   optic.surfaces.add(index=2, z=50, surface_type="plane")
   field = gaussian_field((256, 256), dx=0.004, wavelength=0.0005,
                          waist_radius=0.1)
   train = ScalarOpticalTrain.from_optic(optic)
   output = train.propagate(field)

Planes, units, and normalization
------------------------------------------------------------

``start_surface`` and ``end_surface`` select an inclusive range of surface
indices; the object at index zero cannot be selected.
The input plane is immediately before the first selected vertex, and the output
plane is immediately after the last selected vertex.
Only the gaps between selected vertices propagate.
Neither the object distance nor the last surface's trailing thickness is used.
The selected surfaces remain live references, and edits are revalidated before
propagation without updating the source optic's native material caches.

Use millimeters for spacing, vacuum wavelength, geometry, and vertex positions.
Only material lookup converts wavelength to micrometers.
The field center locates the grid in the surface coordinate plane and is
preserved through screens and propagation.
Amplitudes represent the square root of power density: a lossless interface
does not rescale integrated ``abs(data)**2`` by a refractive-index ratio.
An explicit aperture or phase-profile efficiency can remove power.

Supported prescription
----------------------

Surfaces must be coaxial, transmissive, uncoated, and linked in sequential order.
Native planes, spherical/conic surfaces, and even aspheres are supported.
Native planar thin lenses use quadratic phase with reduced focal parameter
``f``; a collimated beam focuses at ``n_after * f`` in the output medium.
Native constant and radial phase profiles supply phase in radians directly.
A planar ``ImageSurface`` is a marker and cannot change the medium.
The class API documents the exact accepted types and parameter conventions.

Materials require positive real refractive indices and exactly zero extinction,
including in incident and unpropagated trailing media.
Every nonzero catalog extinction value is rejected, however small.
Rotations, decenters, reference coordinate frames, mirrors, coatings,
scattering, GRIN media, negative gaps, and unaudited custom types are rejected.

Only explicit radial, rectangular, or elliptical ``surface.aperture`` objects
clip the field.
``Optic.aperture`` controls ray-launch sampling and does not define a mask here;
provide a physical stop aperture explicitly when needed.

Sampling and derivatives
------------------------

Resolve aperture edges and screen phase gradients, provide enough window for
periodic FFT propagation, and check convergence in spacing and window size.
At nonzero gaps, the propagator discards evanescent content.
Zero gaps apply screens without filtering; differentiable zero gaps use the
identity-preserving decay policy.

NumPy and PyTorch preserve the field's precision and device.
PyTorch supports field and geometry gradients, including finite lens parameters,
phase coefficients, and even-asphere coefficients.
Material indices are scalar metadata without gradients.
Infinite-radius conic bases are constant metadata; polynomial coefficients can
still carry gradients.
Aperture edges and the efficiency square root at zero are nondifferentiable.
