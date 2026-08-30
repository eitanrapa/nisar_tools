"""Geodetic slip inversion from :class:`~nisar_tools.los.LOSStack` displacements.

The rest of the package turns NISAR granules into line-of-sight ground
displacement. This subpackage turns that displacement into slip on a fault: the
observations are adaptively downsampled, a triangulated fault surface is built
from a surface trace, elastic Green's functions relate unit slip on each patch to
the displacement each observation would see, and a regularized, bounded
least-squares solve inverts for the slip distribution.

The design follows `SlipSolve-Curve <https://github.com/x3zou/SlipSolve-Curve>`_
(MATLAB), reimplemented in numpy/scipy. Typical flow::

    from nisar_tools.slip import FaultTrace, FaultMesh, Observations, SlipInversion

    trace = FaultTrace.from_file("fault.kml")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

    obs = Observations.concat([
        Observations.from_los(asc, pair=0, name="asc", frame=frame, fault=trace),
        Observations.from_los(desc, pair=0, name="desc", frame=frame, fault=trace),
    ])

    model = SlipInversion(mesh, obs).solve(smoothing=0.3)

Sign conventions, which matter more here than anywhere else in the package:

* Displacement is **positive toward the sensor**, and the ENU look vector points
  from the target *to* the sensor -- both inherited unchanged from
  :mod:`nisar_tools.geometry`, so the forward model is a plain dot product.
* Depth is negative: the free surface is ``z = 0`` and the fault lies at
  ``z < 0``.
* Positive strike-slip is **left-lateral** (sinistral). Measured, not assumed:
  every element's strike direction is ``cross(eZ, Vnorm)``, and because
  :class:`~nisar_tools.slip.mesh.FaultMesh` winds each element so ``Vnorm`` is
  the trace's *left-hand* normal, that direction works out to exactly
  **minus the trace's tangent**. So positive strike-slip moves the block on the
  trace's left backwards along the trace and the block on its right forwards --
  which, viewed from either side, is left-lateral at any strike.

  The practical consequence: a **right-lateral** fault wants
  ``polarity=(-1, 0, 0)``, constraining strike-slip negative. The San Sebastian
  and Sagaing faults are both right-lateral, which is why the reference
  implementation's Myanmar configuration also pins strike-slip negative.

* Positive dip-slip moves the hanging wall up. Its sign is the one that flips if
  an element is wound the other way, so the mesh pins winding rather than
  policing it after the fact.
"""

from .basis import ElementBasis, NodeBasis
from .covariance import (
    CovarianceModel,
    empirical_variogram,
    fit_covariance,
    observation_covariance,
    scene_covariance,
    whitening_factors,
)
from .diagnostics import (
    cell_size_ladder,
    noise_floor,
    ramp_content,
    ramp_leakage,
    ramp_sweep,
    scene_report,
)
from .edgrn import EdgrnTables, VelocityModel, run_edgrn, write_edgrn_input
from .frame import LocalFrame
from .greens import HalfSpaceTDE
from .inversion import SlipInversion, SlipModel
from .layered import LayeredPointSource
from .layout import MeshLayout, graded_layout, lloyd_layout, triangulate
from .mesh import FaultMesh
from .predict import iterate_sampling, model_rms_min, predicted_los
from .resample import ARCSEC_10, frame_lattice, resample_all, resample_to_frame
from .resolution import (
    adapt_layout,
    element_resolution,
    data_importance,
    resolution_diagonal,
    resolution_gain,
    resolution_kernels,
    resolution_matrix,
    resolution_bias,
    resolution_report,
    posterior_covariance,
    bound_fraction,
    size_field,
    target_element_size,
    weighted_design,
)
from .sampling import Observations
from .surface import FaultSurface, gridfit
from .trace import FaultSegment, FaultTrace

__all__ = [
    "ARCSEC_10",
    "CovarianceModel",
    "EdgrnTables",
    "ElementBasis",
    "FaultMesh",
    "FaultSegment",
    "FaultSurface",
    "FaultTrace",
    "HalfSpaceTDE",
    "LayeredPointSource",
    "LocalFrame",
    "MeshLayout",
    "NodeBasis",
    "Observations",
    "SlipInversion",
    "SlipModel",
    "VelocityModel",
    "adapt_layout",
    "bound_fraction",
    "cell_size_ladder",
    "data_importance",
    "element_resolution",
    "empirical_variogram",
    "fit_covariance",
    "frame_lattice",
    "graded_layout",
    "gridfit",
    "iterate_sampling",
    "lloyd_layout",
    "model_rms_min",
    "noise_floor",
    "observation_covariance",
    "posterior_covariance",
    "predicted_los",
    "ramp_content",
    "ramp_leakage",
    "ramp_sweep",
    "resample_all",
    "resample_to_frame",
    "resolution_bias",
    "resolution_diagonal",
    "resolution_gain",
    "resolution_kernels",
    "resolution_matrix",
    "resolution_report",
    "run_edgrn",
    "scene_covariance",
    "scene_report",
    "size_field",
    "target_element_size",
    "triangulate",
    "weighted_design",
    "whitening_factors",
    "write_edgrn_input",
]
