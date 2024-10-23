#!/usr/bin/env python

import os
import sys
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
from simsopt.configs import get_ncsx_data
from create_windowpanes import create_arbitrary_windowpanes, maximum_coil_radius, clean_components, rotate_windowpane_shapes, planar_vector_list, create_multiple_arbitrary_windowpanes
from helper_functions import rz_surf_to_xyz, rz_normal_to_xyz

nfp = 3
stelsym = True

ncsx_curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(ncsx_curves, currents, nfp, stelsym)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)

curves_to_vtk(curves, 'NCSX', close = True)

ntoroidal = 3
npoloidal = 10

sin_comps_0 = [[0,0.1]]
cos_comps_0 = [[0.1,0]]

for i in range(2):
    windowpanes = create_multiple_arbitrary_windowpanes(ntoroidal, npoloidal, nfp, stelsym, sin_comps_0, cos_comps_0)

windowpane_currents = [Current(1) for i in range(len(windowpanes))]

all_curves = ncsx_curves + windowpanes
all_currents = currents+windowpane_currents
all_coils = coils_via_symmetries(all_curves, all_currents, nfp, stelsym)
all_curves = [c.curve for c in all_coils]
bs_all = BiotSavart(all_coils)

curves_to_vtk(all_curves, 'NCSX+windowpanes', close=True)

NCSX_surface = SurfaceRZFourier.from_vmec_input('input.li383_1.4m')

NCSX_surface.to_vtk('NCSX_surface')

nphi = 32
ntheta = 32
surface_extension = 0.1

winding_surface = SurfaceRZFourier.from_vmec_input('input.li383_1.4m', range="half period", nphi=nphi, ntheta=ntheta)
winding_surface.extend_via_normal(surface_extension)
winding_surface.to_vtk("winding_surface")
def winding_surface_function(tor_angle, pol_angle):
    return rz_surf_to_xyz(winding_surface, tor_angle, pol_angle)


for i in range(2):
    windowpanes = create_multiple_arbitrary_windowpanes(ntoroidal, npoloidal, nfp, stelsym, sin_comps_0, cos_comps_0, winding_surface_function = winding_surface_function)

windowpane_currents = [Current(1) for i in range(len(windowpanes))]

all_curves = ncsx_curves + windowpanes
all_currents = currents+windowpane_currents
all_coils = coils_via_symmetries(all_curves, all_currents, nfp, stelsym)
all_curves = [c.curve for c in all_coils]
bs_all = BiotSavart(all_coils)

curves_to_vtk(all_curves, 'NCSX+windowpanes_wind', close=True)


def winding_surface_function_normal(tor_angle, pol_angle):
    xyz= rz_surf_to_xyz(winding_surface, tor_angle, pol_angle)
    normal_vec = rz_normal_to_xyz(winding_surface, tor_angle, pol_angle)
    return xyz, normal_vec

for i in range(2):
    windowpanes = create_multiple_arbitrary_windowpanes(ntoroidal, npoloidal, nfp, stelsym, sin_comps_0, cos_comps_0, winding_surface_function = winding_surface_function_normal, normaltowinding=True)

windowpane_currents = [Current(1) for i in range(len(windowpanes))]

all_curves = ncsx_curves + windowpanes
all_currents = currents + windowpane_currents
all_coils = coils_via_symmetries(all_curves, all_currents, nfp, stelsym)
all_curves = [c.curve for c in all_coils]
bs_all = BiotSavart(all_coils)

curves_to_vtk(all_curves, 'NCSX+windowpanes_wind_norm', close=True)




############################################

curves_to_vtk(windowpanes, 'NCSX_windowpanes', close=True)

NCSX_surface = SurfaceRZFourier.from_vmec_input('input.li383_1.4m', quadpoints_theta=np.linspace(0,2*np.pi,1000), quadpoints_phi=np.linspace(0,2*np.pi,1000))

NCSX_surface.to_vtk('NCSX_surface_full')
