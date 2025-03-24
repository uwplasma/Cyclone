#!/usr/bin/env python
r"""

In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find windowpane coils that generate a specific target normal field
on a given surface.  In this particular case we consider a vacuum field, so
the target is just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from math import sin, cos
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, CurveXYZFourier)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from V1.create_windowpanes import create_arbitrary_windowpanes, maximum_coil_radius
from V1.V1_helper_functions import rz_surf_to_xyz, rz_normal_to_xyz

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ntoroidalcoils = 3
npoloidalcoils = 12
nfp = 2
stellsym=True

# Major radius of the toroidal surface and of the modular coils
R0 = 1.0

# Minor radius of the toroidal surface and of the modular coils
R1 = 0.5

# Proportion of the maximum possible windowpane coils radius to use for
# radius of the windowpane coils
max_radius_prop = 0.8
coil_radius = max_radius_prop * maximum_coil_radius(ntoroidalcoils, npoloidalcoils, nfp, stellsym, R0=R0, R1=R1)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 2

# Distance to extend the coil winding surface away from the plasma surface
surface_extension = 0.2

# Setting whether the coils' normals are along the normal of the winding
# surface or pointing towards the primary axis
normaltowinding = False

# Choose one option for the coil winding surface

coil_winding_surface = "Default circular torus"
#coil_winding_surface = "'User input' circular torus"
#coil_winding_surface = "Plasma surface"
#coil_winding_surface = "Plasma surface with normal to winding"
#coil_winding_surface = "Extended off of plasma surface"
#coil_winding_surface = "Extended off of plasma surface with normal to winding"
#coil_winding_surface = "User input"
#coil_winding_surface = "User input with normal to winding"

# The sine and cosine components which describe the shape of the planar
# windowpane coils
# For circles:
'''
sin_components = [[coil_radius,0.0]]
cos_components = [[0,coil_radius]]
rotation_angle = 0
'''
# For horizontal ellipses:

sin_components = [[coil_radius,0.0]]
cos_components = [[0,0.5 * coil_radius]]
rotation_angle = 0

# For vertical ellipses:
'''
sin_components = [[0.5 * coil_radius,0.0]]
cos_components = [[0,coil_radius]]
rotation_angle = 0
'''
# For angled ellipses:
'''
sin_components = [[coil_radius,0.0]]
cos_components = [[0,0.5 * coil_radius]]
rotation_angle = np.pi / 4
'''
# For user input shapes:
'''
sin_components = [[coil_radius,0.0],[0.0,0.5*coil_radius]]
cos_components = [[0,0.5*coil_radius]]
rotation_angle = np.pi / 4
'''

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-6)

# Threshold and weight for coil length penalty in the objective function:
LENGTH_THRESHOLD = 2*np.pi*maximum_coil_radius(ntoroidalcoils, npoloidalcoils, nfp, stellsym, R0=R0, R1=R1)
LENGTH_CONSTRAINT_WEIGHT = 0.1

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 100.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / "simsopt" / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

if coil_winding_surface == "Default circular torus":
    winding_surface_function = None
elif coil_winding_surface == "User input circular torus":
    def winding_surface_function(tor_angle, pol_angle):
        return [cos(tor_angle)*cos(pol_angle),sin(tor_angle)*cos(pol_angle),sin(pol_angle)]
elif coil_winding_surface == "Plasma surface":
    s.to_vtk(OUT_DIR + "surf_winding")
    def winding_surface_function(tor_angle, pol_angle):
        return rz_surf_to_xyz(s, tor_angle, pol_angle)
elif coil_winding_surface == "Plasma surface with normal to winding":
    s.to_vtk(OUT_DIR + "surf_winding")
    normaltowinding = True
    def winding_surface_function(tor_angle, pol_angle):
        xyz = rz_surf_to_xyz(s, tor_angle, pol_angle)
        normal_vec = rz_normal_to_xyz(s, tor_angle, pol_angle)
        return xyz, normal_vec
elif coil_winding_surface == "Extended off of plasma surface":
    extended_surface = s
    extended_surface.extend_via_normal(surface_extension)
    extended_surface.to_vtk(OUT_DIR + "surf_winding")
    def winding_surface_function(tor_angle, pol_angle):
        return rz_surf_to_xyz(extended_surface, tor_angle, pol_angle)
elif coil_winding_surface == "Extended off of plasma surface with normal to winding":
    extended_surface = s
    extended_surface.extend_via_normal(surface_extension)
    extended_surface.to_vtk(OUT_DIR + "surf_winding")
    normaltowinding = True
    def winding_surface_function(tor_angle, pol_angle):
        xyz = rz_surf_to_xyz(extended_surface, tor_angle, pol_angle)
        normal_vec = rz_normal_to_xyz(extended_surface, tor_angle, pol_angle)
        return xyz, normal_vec
elif coil_winding_surface == "User input":
    def winding_surface_function(tor_angle, pol_angle):
        xyz = [None, None, None]
        return xyz
elif coil_winding_surface == "User input with normal to winding":
    normaltowinding = True
    def winding_surface_function(tor_angle, pol_angle):
        xyz = [None, None, None]
        normal_vec = [None, None, None]
        return xyz, normal_vec
else:
    winding_surface_function = None

# Create the initial coils:

base_curves = create_arbitrary_windowpanes(ntoroidalcoils, npoloidalcoils, nfp, stellsym, sin_components, cos_components,
                                           rotation_angle=rotation_angle, winding_surface_function = winding_surface_function,
                                           fixed = True, normaltowinding=normaltowinding, order = order)
base_currents = [Current(1e5) for c in base_curves]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ntoroidalcoils*npoloidalcoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]


# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + LENGTH_CONSTRAINT_WEIGHT * sum([QuadraticPenalty(J, LENGTH_THRESHOLD) for J in Jls]) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    
# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
