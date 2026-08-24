# Simple test without pygame
from jps import *
import time
field = [
[1,  1,  1,  1,  1,  1,  1,  1,  1, 1],
[1,  0,  0,  0,  0,  0,  0,  0,  0, 1],
[1,  0,  0,  0,  1,  0,  0,  0,  0, 1],
[1,  0,  0,  0,  1,  1,  1,  0,  0, 1],
[1,  0,  0,  1,  1,  1,  1,  1,  0, 1],
[1,  0,  0,  1,  1,  1,  1,  1,  0, 1],
[1,  0,  0,  1,  0,  0,  0,  0,  0, 1],
[1,  0,  0,  1,  0,  0,  0,  0,  0, 1],
[1,  0,  0,  1,  0,  0,  0,  0, -2, 1],
[1,  0,  0,  1,  0,  0,  0,  0,  0, 1],
[1,  1,  1,  1,  1,  1,  1,  1,  1, 1]]

start_time = time.time()
path = jps(field, 3, 2, 8, 8, 0)
print("short path:", path)
print("full path:", get_full_path(path))
print((time.time()-start_time)*1500*1500)
