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
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil, ToroidalField
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, CurveXYZFourier)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from V1.create_windowpanes import create_multiple_arbitrary_windowpanes, maximum_coil_radius, clean_components, rotate_windowpane_shapes
from V1.V1_helper_functions import rz_surf_to_xyz, rz_normal_to_xyz
from V1.V1_optimization_functions import multiple_sin_cos_components_to_xyz, set_shapes_matrix, set_opt_coil_parameters, change_arbitrary_windowpanes, multiple_change_jacobian

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

# Implement modular coils for toroidal flux
modular = False

# Implement axisymmetric 1/R field for toroidal flux
oneoverR = True
if oneoverR:
    B0 = 1

# Proportion of the maximum possible windowpane coils radius to use for
# radius of the windowpane coils
max_radius_prop = 0.8
coil_radius = max_radius_prop * maximum_coil_radius(ntoroidalcoils, npoloidalcoils, nfp, stellsym, R0=R0, R1=R1)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Distance to extend the coil winding surface away from the plasma surface
surface_extension = 0.1

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

# Set the number of independent shapes to generate
unique_shapes = 5

# User can specify location of shapes (0 - unique_shapes) or where not to place coils (-1)
'''
shapes_matrix = np.array([[0,0,0,1,1,2,2,0,0,0,-1,-1],
                 [1,1,1,2,2,0,0,1,1,1,-1,-1],
                 [2,2,2,0,0,1,1,2,2,2,-1,-1]])
'''

# Otherwise, automatic shapes_matrix creation
try:
    shapes_matrix
except:
    tile = np.array(range(unique_shapes))
    shapes_matrix = np.resize(tile, (ntoroidalcoils, npoloidalcoils))

# Send shapes_matrix to optimization_functions module
set_shapes_matrix(shapes_matrix)

# Templates for the sine and cosine components which describe
# the shape of the planar windowpane coils
# For circles:
'''
sin_components = [[coil_radius,0.0]]
cos_components = [[0,coil_radius]]
rotation_angle = 0
'''
# For horizontal ellipses:
'''
sin_components = [[coil_radius,0.0]]
cos_components = [[0,0.5 * coil_radius]]
rotation_angle = 0
'''
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

# The number of specified shapes
n_specified = 3

sin_components_0 = [[coil_radius,0.0]]
cos_components_0 = [[0,coil_radius]]

sin_components_1 = [[coil_radius,0.0]]
cos_components_1 = [[0,0.5 * coil_radius]]

sin_components_2 = [[coil_radius,0.0]]
cos_components_2 = [[0,0.25*coil_radius]]

#Create rotation angles for shapes
#rotation_angles = [0,np.pi/4,np.pi/3] # User specified
rotation_angles = list(np.zeros(unique_shapes)) # All 0

# Clean the components
sin_components_0 = clean_components(sin_components_0, order)
cos_components_0 = clean_components(cos_components_0, order)

sin_components_1 = clean_components(sin_components_1, order)
cos_components_1 = clean_components(cos_components_1, order)

sin_components_2 = clean_components(sin_components_2, order)
cos_components_2 = clean_components(cos_components_2, order)

sin_components_0, cos_components_0 = rotate_windowpane_shapes(sin_components_0, cos_components_0, rotation_angles[0], order)
sin_components_1, cos_components_1 = rotate_windowpane_shapes(sin_components_1, cos_components_1, rotation_angles[1], order)
sin_components_2, cos_components_2 = rotate_windowpane_shapes(sin_components_2, cos_components_2, rotation_angles[2], order)

# Fill in unspecified shapes

circ_sin = clean_components([[np.max(sin_components_0),0]], order)
circ_cos = clean_components([[0,np.max(cos_components_0)]], order)
circ_comps = np.append(circ_sin.flatten(), circ_cos.flatten())

# Make array of sin and cos components (Comment out/add in lines as necessary for specified shapes)
sin_cos_comps = np.append(sin_components_0.flatten(), cos_components_0.flatten())
sin_cos_comps = np.append(sin_cos_comps, np.append(sin_components_1.flatten(), cos_components_1.flatten()))
sin_cos_comps = np.append(sin_cos_comps, np.append(sin_components_2.flatten(), cos_components_2.flatten()))
for i in range(unique_shapes-n_specified):
    sin_cos_comps = np.append(sin_cos_comps, circ_comps)

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
    extended_surface = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
    extended_surface.extend_via_normal(surface_extension)
    extended_surface.to_vtk(OUT_DIR + "surf_winding")
    def winding_surface_function(tor_angle, pol_angle):
        return rz_surf_to_xyz(extended_surface, tor_angle, pol_angle)
elif coil_winding_surface == "Extended off of plasma surface with normal to winding":
    extended_surface = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
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
# For some reason, the first initialized coils
# don't work. So I initialize them twice here:

for i in range(2):
    base_curves = create_multiple_arbitrary_windowpanes(ntoroidalcoils, npoloidalcoils, nfp, stellsym, sin_components_0, cos_components_0,
                                                        unique_shapes = unique_shapes, shapes_matrix = shapes_matrix,
                                                        rotation_angles=rotation_angles,
                                                        winding_surface_function = winding_surface_function,
                                                        fixed = True, normaltowinding=normaltowinding, order = order,
                                                        sin_components_1 = sin_components_1, cos_components_1 = cos_components_1,
                                                        sin_components_2 = sin_components_2, cos_components_2 = cos_components_2)

if modular:
    for i in range(ntoroidalcoils):
        curve = CurveXYZFourier(75, order=1)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcoils)
        curve.set("xc(0)", cos(angle)*R0)
        curve.set("xc(1)", cos(angle)*(R1+coil_radius))
        curve.set("yc(0)", sin(angle)*R0)
        curve.set("yc(1)", sin(angle)*(R1+coil_radius))
        # The the next line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
        curve.set("zs(1)", -(R1+coil_radius))
        curve.x = curve.x  # need to do this to transfer data to C++
        curve.fix_all()
        base_curves.append(curve)

base_currents = [Current(1)*1e5 for c in base_curves]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
#base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

if oneoverR:
    tf = ToroidalField(R0, B0)
    bs = bs + tf

bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Send rest of coil parameters to optimization_functions module
set_opt_coil_parameters(ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes, winding_surface_function, order, curves, len(base_currents))

# Create a jacobian object
multiple_change_jacobian = multiple_change_jacobian()

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
    JF.x = list(dofs[:len(base_currents)]) + list(JF.x[len(base_currents):])
    dofsarr = jnp.array(dofs[len(base_currents):])
    curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)
    change_arbitrary_windowpanes(curves, curves_xyz_dofs)
    dofs_grad = JF.dJ()
    jacobian = multiple_change_jacobian(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)
    grad = jnp.matmul(dofs_grad[len(base_currents):].reshape((1,len(dofs_grad)-len(base_currents))), jacobian)
    grad = jnp.append(jnp.array(dofs_grad[:len(base_currents)]), grad)
    J = JF.J()
    print(J)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
current_dofs = np.array([c.x for c in base_currents])
dofs = np.append(current_dofs, sin_cos_comps)
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

end_dofs = res.x
JF.x = list(end_dofs[:len(base_currents)]) + list(JF.x[len(base_currents):])
dofsarr = jnp.array(end_dofs[len(base_currents):])
curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                                     unique_shapes, winding_surface_function = winding_surface_function, order=order)
change_arbitrary_windowpanes(curves, curves_xyz_dofs)

curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
