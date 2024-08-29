#!/usr/bin/env python
r"""

Update this

The purpose of this code is to test my cleaned up code for making windowpane
in Simsopt and confirm that the code functions as needed by recreating the
'stage_two_optimization' example file using the windowpane functions.

From 'stage_two_optimization':
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

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

import pandas as pd
import os
from pathlib import Path
import numpy as np
from math import sin, cos
import jax.numpy as jnp
import jax
from functools import partial
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil, ToroidalField
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, CurveXYZFourier)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from create_windowpanes import create_arbitrary_windowpanes, maximum_coil_radius, clean_components, rotate_windowpane_shapes, planar_vector_list, create_multiple_arbitrary_windowpanes
from helper_functions import rz_surf_to_xyz, rz_normal_to_xyz
from optimization_functions import set_shapes_matrix, multiple_sin_cos_components_to_xyz, multiple_change_jacobian, change_arbitrary_windowpanes
from verification import set_JF, set_coil_parameters, find_sum_planar_error_mult_epsilon

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ntoroidalcoils = 10
npoloidalcoils = 15
nfp = 2
stellsym=True

# The number of windings for each windowpane coil
n_windings = 40

modular = False
oneoverR = True
B0 = 1

# Major radius of the toroidal surface and of the modular coils
R0 = 1.0

# Minor radius of the toroidal surface and of the modular coils
R1 = 0.5

# Proportion of the maximum possible windowpane coils radius to use for
# radius of the windowpane coils
max_radius_prop = 1.0
coil_radius = max_radius_prop * maximum_coil_radius(ntoroidalcoils, npoloidalcoils, nfp, stellsym, R0=R0, R1=R1)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 3

# Distance to extend the coil winding surface away from the plasma surface
surface_extension = 0.1

# Setting whether the coils' normals are along the normal of the winding
# surface or pointing towards the primary axis
normaltowinding = False

# The number of initial conditions to randomly sample
sample_size = 1

# Choose one option for the coil winding surface

#coil_winding_surface = "Default circular torus"
#coil_winding_surface = "'User input' circular torus"
#coil_winding_surface = "Plasma surface"
#coil_winding_surface = "Plasma surface with normal to winding"
coil_winding_surface = "Extended off of plasma surface"
#coil_winding_surface = "Extended off of plasma surface with normal to winding"
#coil_winding_surface = "User input"
#coil_winding_surface = "User input with normal to winding"

unique_shapes = 6
'''
shapes_matrix = np.array([[0,0,0,1,1,2,2,0,0,0,-1,-1],
                 [1,1,1,2,2,0,0,1,1,1,-1,-1],
                 [2,2,2,0,0,1,1,2,2,2,-1,-1]])
'''

shapes = np.append(-1, np.arange(unique_shapes))
shapes_matrix = np.random.choice(shapes, (ntoroidalcoils, npoloidalcoils))

print(shapes_matrix)

try:
    shapes_matrix
except:
    tile = np.array(range(unique_shapes))
    shapes_matrix = np.resize(tile, (ntoroidalcoils, npoloidalcoils))
#print(shapes_matrix)
# The sine and cosine components which describe the shape of the planar
# windowpane coils
# For circles:

sin_components_0 = [[coil_radius,0.0]]
cos_components_0 = [[0,coil_radius]]
rotation_angle = 0
#print(coil_radius)
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

sin_components_1 = [[coil_radius,0.0]]
cos_components_1 = [[0,0.5 * coil_radius]]
rotation_angle = np.pi / 4

# For user input shapes:

#sin_components_2 = [[coil_radius,0.0],[0.0,0.5*coil_radius]]
sin_components_2 = [[coil_radius,0.0]]
cos_components_2 = [[0,0.25*coil_radius]]
rotation_angle = np.pi / 4

rotation_angles = [0,np.pi/4,np.pi/3]
rotation_angles = list(np.zeros(unique_shapes))
# Clean the components
sin_components_0 = clean_components(sin_components_0, order)
cos_components_0 = clean_components(cos_components_0, order)

sin_components_1 = clean_components(sin_components_1, order)
cos_components_1 = clean_components(cos_components_1, order)

sin_components_2 = clean_components(sin_components_2, order)
cos_components_2 = clean_components(cos_components_2, order)

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-6)

# Threshold and weight for coil length penalty in the objective function:
LENGTH_THRESHOLD = 2*np.pi*maximum_coil_radius(ntoroidalcoils, npoloidalcoils, nfp, stellsym, R0=R0, R1=R1)
LENGTH_CONSTRAINT_WEIGHT = 0.1

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.03
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
MAXITER = 400 # if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent).resolve()
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
                                               unique_shapes = unique_shapes, shapes_matrix = shapes_matrix, rotation_angles=rotation_angles,
                                               winding_surface_function = winding_surface_function,
                                               fixed = True, normaltowinding=normaltowinding, order = order, sin_components_1 = sin_components_1,
                                               cos_components_1 = cos_components_1, sin_components_2 = sin_components_2, cos_components_2 = cos_components_2)

modular_curves = []
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


#base_currents = [Current(1)*1e5 for c in base_curves]
'''
base_currents = []
mod_currents = []
real_currents = []
for i in range(ntoroidalcoils):
    pol_current = Current(1)*1e5
    real_currents.append(pol_current)
    for j in range(npoloidalcoils):
        if j < 10:
            base_currents.append(pol_current)
    mod_currents.append(pol_current / n_windings)
for curr in mod_currents:
    base_currents.append(curr)
'''

base_currents = [Current(0.0001)*1e9 for c in base_curves]

# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:

#for i in range(len(base_currents)):
#    base_currents[i].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

if oneoverR:
    tf = ToroidalField(R0, B0)
    bs = bs + tf

bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")

#pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}

# B*n / B

Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n": np.sum(Bbs * s.unitnormal(), axis=2)[:, :, None], "B.n/B": BdotN[:, :, None], "B": Bmod}

s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs, definition='local')
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
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) #\
#    + LENGTH_CONSTRAINT_WEIGHT * sum([QuadraticPenalty(J, LENGTH_THRESHOLD) for J in Jls]) #\
#    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) #\
#    + CS_WEIGHT * Jcsdist #\

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

current_dofs = np.array([c.x for c in base_currents])
sin_components_0, cos_components_0 = rotate_windowpane_shapes(sin_components_0, cos_components_0, rotation_angles[0], order)
sin_components_1, cos_components_1 = rotate_windowpane_shapes(sin_components_1, cos_components_1, rotation_angles[1], order)
sin_components_2, cos_components_2 = rotate_windowpane_shapes(sin_components_2, cos_components_2, rotation_angles[2], order)
sin_cos_comps = np.append(sin_components_0.flatten(), cos_components_0.flatten())
sin_cos_comps = np.append(sin_cos_comps, np.append(sin_components_1.flatten(), cos_components_1.flatten()))
sin_cos_comps = np.append(sin_cos_comps, np.append(sin_components_2.flatten(), cos_components_2.flatten()))
circ_sin = clean_components([[np.max(sin_components_0),0]], order)
circ_cos = clean_components([[0,np.max(cos_components_0)]], order)
circ_comps = np.append(circ_sin.flatten(), circ_cos.flatten())
for i in range(unique_shapes-3):
    sin_cos_comps = np.append(sin_cos_comps, circ_comps)

dofs = list(np.append(current_dofs, sin_cos_comps))

set_JF(JF)
set_shapes_matrix(shapes_matrix)
set_coil_parameters(ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes, winding_surface_function, order, curves, len(base_currents))

best_start_dofs = dofs

dofsarr = jnp.array(dofs[len(base_currents):])
curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)
change_arbitrary_windowpanes(curves, curves_xyz_dofs)
multiple_change_jacobian = multiple_change_jacobian()
jacobian = multiple_change_jacobian(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)

df = pd.DataFrame(jacobian)
df.to_excel('output/test.xlsx', index=False)

init_dofs = list(dofs)

#error_vec = find_sum_planar_error_mult_epsilon(init_dofs, first_epsilon = 0.0001, num_epsilon = 5, printout=True, graph=False, save=True)

#print("Error vec:")
#print(error_vec)

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
#res = minimize(fun, dofs, jac=True)

end_dofs = res.x

#print('end')
print(end_dofs)
#fun(end_dofs)

JF.x = list(end_dofs[:len(base_currents)]) + list(JF.x[len(base_currents):])

dofsarr = jnp.array(end_dofs[len(base_currents):])
curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, 2, True,
                                                     unique_shapes, winding_surface_function = winding_surface_function, order=order)
change_arbitrary_windowpanes(curves, curves_xyz_dofs)



#J_best,_ = fun(dofs)
#best_dofs = dofs
'''
for i in range(sample_size):
    print(i)
    for c in real_currents:
        c.x = [1.0]
    current_dofs = np.array([c.x for c in real_currents])
    start_dofs = list(np.append(current_dofs, np.random.uniform(-0.2, 0.2, len(dofs)-len(real_currents))))
    dofs = start_dofs
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    J,_ = fun(dofs)
    if J < J_best:
        J_best = J
        best_dofs = dofs
        best_start_dofs = start_dofs
'''


#print("Best J is:")
#fun(best_dofs)
#print("With dofs:")
#print(best_dofs)
#print("Starting at:")
#print(best_start_dofs)


#curves_to_vtk() for start_dofs
curves_to_vtk(curves, OUT_DIR + "curves_opt_short")

Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n": np.sum(Bbs * s.unitnormal(), axis=2)[:, :, None], "B.n/B": BdotN[:, :, None], "B": Bmod}

s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")

os.system("cp output/ ../../../../../mnt/c/Users/dseid/Downloads/ -r")
