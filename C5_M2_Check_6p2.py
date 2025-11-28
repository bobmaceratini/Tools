import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *
from RMAttitudeStaticDeterminationTools import *

I1 = 100
I2 = 150
I3 = 200

sigma_BN = np.array([0.5,0.5,0.3])
sigma_GB = np.array([-0.2,0.3,0.1])
w_BN_G = np.array([0.1,0.2,0.3])

L_N = np.array([1,1,1])

IB = np.array([[I1,0,0],[0, I2, 0],[0,0,I3]])
IB_inv = np.linalg.inv(IB)

BN = MRP2DCM(sigma_BN)
GB = MRP2DCM(sigma_GB)

L_B = BN @ L_N
w_BN_B = GB.T @ w_BN_G

w_BN_B_tilde = tilde(w_BN_B)

w_BN_B_dot = IB_inv @ (- w_BN_B_tilde @ IB @ w_BN_B + L_B)

print(f"Omega time derivative:\n {w_BN_B_dot}")

#doucle check:
