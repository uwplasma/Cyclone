import os
import time
import numpy as np
import sys
from V1.V1_helper_functions import generate_permutations, save_scaled_iota

num_iter = 30

# The variable list is:
# filename, ntoroidalcoils, npoloidalcoils, ntoroidalcoils_TF, R0, R1, R1_TF, surface_extension, unique_shapes, MAXITER

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

input_dir = 'input/'
init_stel = 'precise QA'

line_list = generate_permutations(('ntoroidalcoils', 10, 11, 1),
                                  ('npoloidalcoils', 15, 16, 1),
                                  ('ntoroidalcoils_TF', 10, 11, 1),
                                  ('R0', 1, 1.1, 1),
                                  ('R1', 0.3, 0.4, 1),
                                  ('R1_TF', 0.4, 0.5, 1),
                                  ('surface_extension', 0.1, 0.2, 1),
                                  ('unique_shapes', 55, 56, 15),
                                  ('MAXITER', 400, 401, 100))

if file:
    save_scaled_iota(init_stel, out_dir = input_dir, r=0.05)
    for file in os.listdir(input_dir):
        for in_line in line_list:
            line = input_dir + file + ' ' + str(in_line).replace('(','').replace(')','').replace("'",'').replace(',','')
            for i in range(num_iter):
                while int(os.popen('ps -ef | grep optimization.py | wc -l').read()) - 2 > 5:
                    time.sleep(10)
                os.system('python3 optimization.py {} &'.format(line))
                time.sleep(3)

else:
    for in_line in line_list:
        line = str(in_line).replace('(','').replace(')','').replace("'",'').replace(',','')
        for i in range(num_iter):
            while int(os.popen('ps -ef | grep optimization.py | wc -l').read()) - 2 > 5:
                time.sleep(10)
            os.system('python3 optimization.py {} &'.format(line))
            time.sleep(3)
