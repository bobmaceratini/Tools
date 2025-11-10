import numpy as np
from RMKinematicTools import *

s=np.array([0.1,0.2,0.3])
ss = MRP2Shadow(s)

print("s:", s)
print("Shadow ss:", ss)
