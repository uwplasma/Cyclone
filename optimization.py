import numpy as np
from read_in import read_in_toml_config
from optimization_functions import generate_dofs_from_dictionary, run_minimize
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import curves_to_vtk, CurveLength, CurveCurveDistance, CurveSurfaceDistance, MeanSquaredCurvature, LinkingNumber, LpCurveCurvature
import sys
import os

# Export information
out_dir = 'output/'
os.makedirs(out_dir, exist_ok=True)

# Read in coil info and generate dofs
config_file = sys.argv[1]
full_dictionary = read_in_toml_config(config_file)
full_dofs = generate_dofs_from_dictionary(full_dictionary)

import tomli
with open(config_file, mode="rb") as fp:
    config = tomli.load(fp)
MAXITER = config['optimization']['MAXITER']

# Export initial coil configuration
coils = full_dictionary['all_coils']
curves = [c.curve for c in coils]

# Could theoretically have a problem with race conditions, but shouldn't be a major issue
file_index=0
while out_dir+f'optimization_init_{file_index}.vtu' in os.listdir():
    file_index += 1

curves_to_vtk(curves, out_dir+f'optimization_init_{file_index}')

# Create objective function
Jls = [CurveLength(c) for c in full_dictionary['windowpane_coils']['curves']]
init_lengths = np.copy([jls.J() for jls in Jls])
Jccdist = CurveCurveDistance(curves, 0.05)
Jcs = sum([LpCurveCurvature(c, 2, 6*np.pi / init_lengths[i]) for i, c in enumerate(full_dictionary['windowpane_coils']['curves'])])
JL = sum(QuadraticPenalty(Jls[i], 1.5*init_lengths[i], "max") for i in range(len(Jls)))
linknum = LinkingNumber(curves)

#Jls = sum([CurveLength(c) for c in curves])
#Jcs = CurveSurfaceDistance(curves, full_dictionary['magnetic_surface'], 0.05)
magnetic_surface = full_dictionary['magnetic_surface']
objective = SquaredFlux(magnetic_surface, full_dictionary['biot_savart_field'], definition='local')
objective = objective + 0.01 * JL + linknum# + 0.1 * Jccdist + 100# * Jp

# Run optimization
full_dofs = run_minimize(full_dofs, full_dictionary, objective, options = {'maxiter' : MAXITER})

# Export final coil configuration
curves_to_vtk(curves, out_dir+f'optimization_coils_{file_index}')

# Export final plasma surface configuration
nphi = 61
ntheta = 62
bs = full_dictionary['biot_savart_field']

Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.sum(Bbs * magnetic_surface.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n": np.sum(Bbs * magnetic_surface.unitnormal(), axis=2)[:, :, None], "B.n/B": BdotN[:, :, None], "B": Bmod}

magnetic_surface.to_vtk(out_dir+f'optimization_surf_{file_index}', extra_data=pointData)

# Calculating output metrics for export
last_obj = objective.J()
max_BdotN = np.max(np.abs(BdotN))
meanabs_BdotN = np.mean(np.abs(BdotN))
RMS_BdotN = (np.mean(BdotN*BdotN))**0.5

currents = full_dictionary['all_currents']

max_current = max([c.get_value() for c in currents])
max_curvature = max([np.max(c.kappa()) for c in curves])

# Finding input parameters for export

simsopt_current_value = config['optimization']['simsopt_current_value']
magnetic_surface = config['optimization']['magnetic_surface']

stellarator_default_current = config['stellarator_coils']['default_current']
stellarator_planar_flag = config['stellarator_coils']['planar_flag']
stellarator_axis_representation = config['stellarator_coils']['axis_representation']
stellarator_ncurves = config['stellarator_coils']['ncurves']
if isinstance(stellarator_ncurves, (list, np.ndarray, tuple)):
    stellarator_ncurves = (f"{stellarator_ncurves}".replace(" ", "")).replace(",", "/")
stellarator_unique_shapes = config['stellarator_coils']['unique_shapes']
stellarator_tile_as = config['stellarator_coils']['tile_as']
if isinstance(stellarator_tile_as, (list, np.ndarray, tuple)):
    stellarator_tile_as = (f"{stellarator_tile_as}".replace(" ", "")).replace(",", "/")
stellarator_R0 = config['stellarator_coils']['R0']
stellarator_R1 = config['stellarator_coils']['R1']
stellarator_order = config['stellarator_coils']['order']
stellarator_fixed = config['stellarator_coils']['fixed']
if isinstance(stellarator_fixed, (list, np.ndarray, tuple)):
    stellarator_fixed = (f"{stellarator_fixed}".replace(" ", "")).replace(",", "/")
stellarator_rotation_opt_flag = config['stellarator_coils']['optimizables']['rotation_opt_flag']
stellarator_normal_opt_flag = config['stellarator_coils']['optimizables']['normal_opt_flag']
stellarator_center_opt_flag = config['stellarator_coils']['optimizables']['center_opt_flag']
stellarator_center_opt_type_flag = config['stellarator_coils']['optimizables']['center_opt_type_flag']
stellarator_planar_opt_flag = config['stellarator_coils']['optimizables']['planar_opt_flag']
stellarator_nonplanar_opt_flag = config['stellarator_coils']['optimizables']['nonplanar_opt_flag']

windowpane_default_current = config['windowpane_coils']['default_current']
windowpane_planar_flag = config['windowpane_coils']['planar_flag']
windowpane_surface_representation = config['windowpane_coils']['surface_representation']
windowpane_normal_to_winding = config['windowpane_coils']['normal_to_winding']
windowpane_surface_extension = config['windowpane_coils']['surface_extension']
windowpane_ntoroidalcurves = config['windowpane_coils']['ntoroidalcurves']
if isinstance(windowpane_ntoroidalcurves, (list, np.ndarray, tuple)):
    windowpane_ntoroidalcurves = (f"{windowpane_ntoroidalcurves}".replace(" ", "")).replace(",", "/")
windowpane_npoloidalcurves = config['windowpane_coils']['npoloidalcurves']
if isinstance(windowpane_npoloidalcurves, (list, np.ndarray, tuple)):
    windowpane_npoloidalcurves = (f"{windowpane_npoloidalcurves}".replace(" ", "")).replace(",", "/")
windowpane_unique_shapes = config['windowpane_coils']['unique_shapes']
windowpane_tile_as = config['windowpane_coils']['tile_as']
if isinstance(windowpane_tile_as, (list, np.ndarray, tuple)):
    windowpane_tile_as = (f"{windowpane_tile_as}".replace(" ", "")).replace(",", "/")
windowpane_R0 = config['windowpane_coils']['R0']
windowpane_R1 = config['windowpane_coils']['R1']
windowpane_coil_radius = config['windowpane_coils']['coil_radius']
windowpane_order = config['windowpane_coils']['order']
windowpane_fixed = config['windowpane_coils']['fixed']
if isinstance(windowpane_fixed, (list, np.ndarray, tuple)):
    windowpane_fixed = (f"{windowpane_fixed}".replace(" ", "")).replace(",", "/")
windowpane_rotation_opt_flag = config['windowpane_coils']['optimizables']['rotation_opt_flag']
windowpane_normal_opt_flag = config['windowpane_coils']['optimizables']['normal_opt_flag']
windowpane_center_opt_flag = config['windowpane_coils']['optimizables']['center_opt_flag']
windowpane_center_opt_type_flag = config['windowpane_coils']['optimizables']['center_opt_type_flag']
windowpane_planar_opt_flag = config['windowpane_coils']['optimizables']['planar_opt_flag']
windowpane_nonplanar_opt_flag = config['windowpane_coils']['optimizables']['nonplanar_opt_flag']

# Exporting to stats_info file
write_line = f"{config_file},{magnetic_surface},{simsopt_current_value},{MAXITER},{stellarator_default_current},{stellarator_planar_flag},{stellarator_axis_representation},{stellarator_ncurves},{stellarator_unique_shapes},{stellarator_tile_as},{stellarator_R0},{stellarator_R1},{stellarator_order},{stellarator_fixed},{stellarator_rotation_opt_flag},{stellarator_normal_opt_flag},{stellarator_center_opt_flag},{stellarator_center_opt_type_flag},{stellarator_planar_opt_flag},{stellarator_nonplanar_opt_flag},{windowpane_default_current},{windowpane_planar_flag},{windowpane_surface_representation},{windowpane_normal_to_winding},{windowpane_surface_extension},{windowpane_ntoroidalcurves},{windowpane_npoloidalcurves},{windowpane_unique_shapes},{windowpane_tile_as},{windowpane_R0},{windowpane_R1},{windowpane_coil_radius},{windowpane_order},{windowpane_fixed},{windowpane_rotation_opt_flag},{windowpane_normal_opt_flag},{windowpane_center_opt_flag},{windowpane_center_opt_type_flag},{windowpane_planar_opt_flag},{windowpane_nonplanar_opt_flag},{last_obj},{max_BdotN},{meanabs_BdotN},{RMS_BdotN},{max_current},{max_curvature}"

with open('stats_info.csv', 'a') as f:
    f.write(write_line)
    f.write('\n')