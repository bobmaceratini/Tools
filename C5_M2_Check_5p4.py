import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *
from RMAttitudeStaticDeterminationTools import *

I1 = 1000
I2 = 1500
I3 = 2000
m = 10
r_CO_N = np.array([10,15,20])

IB = np.array([[I1,0,0],[0, I2, 0],[0,0,I3]])

sigma_BN = np.array([0.1,0.2,0.3])
BN = MRP2DCM(sigma_BN)

r_CO_B = -BN @ r_CO_N

R_CO_B_tilde = tilde(r_CO_B)
I_O_B = IB + m* R_CO_B_tilde @ R_CO_B_tilde.T
print(f"Inertia tensor in B body frame:\n {I_O_B}")

#doucle check:
