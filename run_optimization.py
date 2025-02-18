import os
import time
import numpy as np
import sys
from helper_functions import generate_permutations, create_config_from_input_line, save_scaled_iota

num_iter = 30

input_dir = 'input/'
init_stel = 'precise QA'
base_config = 'statistics_base_config.toml'
config_dir = 'config_dir_'

for arg in sys.argv:
    if 'file' in arg:
        file = arg.replace('file:','')
        if file.lower() == 'true':
            file = True
        elif file.lower() == 'false':
            file = False
        else:
            print('Invalid bool specified for file')

try:
    file
except:
    file = False

os.makedirs(input_dir, exist_ok=True)

magnetic_surf_list = ['input.LandremanPaul2021_QA']

if file:
    save_scaled_iota(init_stel, out_dir = input_dir, r=0.05)
    os.system('bash fix_qsc.sh {}input.{}*'.format(input_dir, init_stel.replace(' ','_')))
    magnetic_surf_list = [file for file in os.listdir(f'{input_dir}')]

line_list = generate_permutations(('magnetic_surface', 'file', magnetic_surf_list),
                                  ('simsopt_current_value', float, 1e-4, 1.001e-4, 1e-4),
                                  ('MAXITER', int, 100, 601, 100),
                                  ('stellarator_default_current', float, 1e5, 1.001e5, 1e5),
                                  ('stellarator_planar_flag', bool, [True, False]),
                                  ('stellarator_axis_representation', 'file', ['simsopt/tests/test_files/wout_LandremanPaul2021_QA_lowres.nc']),
                                  ('stellarator_ncurves', int, 3, 3.5 ,1),
                                  ('stellarator_unique_shapes', int, 1, 1.5, 1),
                                  ('stellarator_tile_as', list, ['random']),
                                  ('stellarator_R0', float, 1, 1.2, 0.1),
                                  ('stellarator_R1', float, 0.5, 0.6, 1),
                                  ('stellarator_order', int, 3, 3.5, 1),
                                  ('stellarator_fixed', list, [[]]),
                                  ('stellarator_rotation_opt_flag', bool, [True]),
                                  ('stellarator_normal_opt_flag', bool, [True]),
                                  ('stellarator_center_opt_flag', bool, [True]),
                                  ('stellarator_center_opt_type_flag', list, ['direct']),
                                  ('stellarator_planar_opt_flag', bool, [True]),
                                  ('stellarator_nonplanar_opt_flag', bool, [False]),
                                  ('windowpane_default_current', float, 1e3, 1.001e3, 1e3),
                                  ('windowpane_planar_flag', bool, [True, False]),
                                  ('windowpane_surface_representation', 'file', ['simsopt/tests/test_files/wout_LandremanPaul2021_QA_lowres.nc']),
                                  ('windowpane_normal_to_winding', bool, [True]),
                                  ('windowpane_surface_extension', float, 0.2, 0.3, 0.2),
                                  ('windowpane_ntoroidalcurves', int, 4, 8, 1),
                                  ('windowpane_npoloidalcurves', int, 4, 8, 1),
                                  ('windowpane_unique_shapes', int, 8, 32, 1),
                                  ('windowpane_tile_as', list, ['random']),
                                  ('windowpane_R0', float, 1, 1.2, 1),
                                  ('windowpane_R1', float, 0.6, 0.7, 1),
                                  ('windowpane_coil_radius', float, 0.04, 0.08, 0.01),
                                  ('windowpane_order', int, 3, 6, 5),
                                  ('windowpane_fixed', list, [[]]),
                                  ('windowpane_rotation_opt_flag', bool, [True]),
                                  ('windowpane_normal_opt_flag', bool, [True]),
                                  ('windowpane_center_opt_flag', bool, [True]),
                                  ('windowpane_center_opt_type_flag', list, ['direct']),
                                  ('windowpane_planar_opt_flag', bool, [True]),
                                  ('windowpane_nonplanar_opt_flag', bool, [True, False])
                                  )

i=0
while config_dir+f'{i}' in os.listdir(input_dir):
    i += 1
config_dir = config_dir+f'{i}'
os.makedirs(input_dir+config_dir)

for in_line in line_list:
    create_config_from_input_line(base_config, in_line, out_dir=input_dir+config_dir)

for config_file in os.listdir(f"{input_dir}{config_dir}"):
    for i in range(num_iter):
        while int(os.popen('ps -ef | grep optimization.py | wc -l').read()) - 2 > 5:
            time.sleep(10)
        os.system('python3 optimization.py {} &'.format(input_dir+config_dir+"/"+config_file))
        time.sleep(3)