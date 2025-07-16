import numpy as np
from read_in import read_in_toml_config
from optimization_functions import generate_dofs_from_dictionary, run_minimize
from simsopt.objectives import SquaredFlux
from simsopt.geo import curves_to_vtk, CurveCurveDistance, LinkingNumber, LpCurveCurvature
import os

# Create output folder
out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

# Import config file
config_file = 'examples/example_toml.toml'
full_dictionary = read_in_toml_config(config_file)
full_dofs = generate_dofs_from_dictionary(full_dictionary)

# Export initial coil configuration
coils = full_dictionary['all_coils']
curves = [c.curve for c in coils]
curves_to_vtk(curves, f'{out_dir}/init_curves')

# Create objective function
magnetic_surface = full_dictionary['magnetic_surface']
objective = SquaredFlux(magnetic_surface, full_dictionary['biot_savart_field'], definition='local')
linknum = LinkingNumber(curves)
Jcc = CurveCurveDistance(curves, 0.1)

windowpane_curves = full_dictionary['windowpane_coils']['curves']

Jkwindowpane = sum([LpCurveCurvature(c, 2, 6000) for i, c in enumerate(windowpane_curves)])


objective = objective + linknum + Jcc + 0.1 * Jkwindowpane

# Run minimization
full_dofs = run_minimize(full_dofs, full_dictionary, objective)

# Export optimized coils
curves_to_vtk(curves, f'{out_dir}/post_curves', close=True)

# Export surface
bs = full_dictionary['biot_savart_field']

nphi = 61
ntheta = 62
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.sum(Bbs * magnetic_surface.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n": np.sum(Bbs * magnetic_surface.unitnormal(), axis=2)[:, :, None], "B.n/B": BdotN[:, :, None], "B": Bmod}

magnetic_surface.to_vtk(f'{out_dir}/surface', extra_data=pointData)
