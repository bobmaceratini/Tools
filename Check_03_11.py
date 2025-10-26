import numpy as np
from RMKinematicTools import *

s = np.array([0.1,0.2,0.3])
DCM = CRP2DCM(s)
s_check = DCM2CRP(DCM)
print("Classical Rodrigues Parameters:", s)
print("Direction Cosine Matrix from CRP:\n", DCM)
print("CRP check from Direction Cosine Matrix:", s_check)

DCM = np.array([[0.333333, -0.666667, 0.666667 ],
                [0.871795 ,  0.487179,  0.0512821 ],  
                [-0.358974, 0.564103,  0.74359]])

s_back = DCM2CRP(DCM)
DCM_check = CRP2DCM(s_back)
print("\nDirection Cosine Matrix:\n", DCM)
print("Classical Rodrigues Parameters from DCM:", s_back)
print("DCM check from CRP:\n", DCM_check)

sBN=np.array([-0.3, 0.3, 0.1])
sFN=np.array([0.1, 0.2, 0.3])
sBF =1/(1 + np.dot(sBN, sFN)) * (sBN - sFN + np.cross(sBN, sFN))
print("\nCRP from frame B to N:", sBN)
print("CRP from frame F to N:", sFN)
print("CRP from frame B to F:", sBF)
