import os
import time

num_iter = 30

first_unique = 20
last_unique = 80
step_size = 5
ran = int(((last_unique - first_unique) / step_size) + 1)
print(ran)

for num_shapes in range(ran):
    for i in range(num_iter):
        while int(os.popen('ps -ef | grep optimization.py | wc -l').read()) - 2 > 5:
            time.sleep(10)
        num_shapes_ = first_unique + num_shapes * step_size
        os.system('python3 optimization.py {} &'.format(num_shapes_))
        time.sleep(3)
