import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *


I1 = 100
I2 = 150
I3 = 200
IB = np.array([[I1,0,0],[0, I2, 0],[0,0,I3]])

sigma_BN = np.array([0.1,0.2,0.3])
BN = MRP2DCM(sigma_BN)

IN = BN.T @ IB @ BN
print(f"Inertia tensor in N reference frame:\n {IN}")