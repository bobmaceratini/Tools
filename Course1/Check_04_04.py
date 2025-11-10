import numpy as np
from RMKinematicTools import *
from RMAttitudeStaticDeterminationTools import *

v1b = np.array([0.8273, 0.5541, -0.0920])
v2b = np.array([-0.8285, 0.5522, -0.0955])

v1n = np.array([-0.1517, -0.9669, 0.2050])
v2n = np.array([-0.8393, 0.4494, -0.3044])

VB = [v1b, v2b]
VN = [v1n, v2n]

q, eig = Davenport2Quat(VB, VN, weights=[1, 1])
print("Estimated Quaternion from Davenport's Q-Method:")    
print(q)
BN_est = Quat2DCM(q)
print("Estimated Direction Cosine Matrix (DCM):")
print(BN_est)

DCM = Triad2DCM(v1b, v2b, v1n, v2n)
print("Direction Cosine Matrix (DCM) from TRIAD method:")
print(DCM)  

print(f"eigenvalue from Davenport's Q-Method:", eig)

q, eig = QUEST2Quat(VB, VN, weights=[1, 1])
print("Estimated Quaternion from QUEST method:")
print(q)
DCM = Quat2DCM(q)
print("Estimated Direction Cosine Matrix (DCM) from QUEST method:")
print(DCM)
print(f"eigenvalue from QUEST method:", eig)

q = OLEA2Quat(VB,VN,weights=[1, 1])
DCM = Quat2DCM(q)
print(f"Estimated Quaterion from OLEA",q)
print(f"DCM from OLEA ",DCM)
