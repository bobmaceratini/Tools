import numpy as np  
from RMKinematicTools import *
print("")
print("----- Assignment 1 -----")
qBN = np.array([0.774597,0.258199,0.516398,0.258199])
qFB = np.array([0.359211,0.898027,0.179605,0.179605])
qFN = QuatSum(qBN, qFB)
qFB = QuatSub(qFN,qBN)
print("Quaternion qBN:", qBN)
print("Quaternion qFB:", qFB)   

print("Quaternion qFN1:", qFN)
pFNf = QuatFlip(qFN)
print("Flipped Quaternion qFNf:", pFNf)

print("")
print("----- Assignment 2 -----")
qFN =np.array([0.359211,0.898027,0.179605,0.179605])
qBN =np.array([-0.377964,0.755929,0.377964,0.377964])
qFB = QuatSub(qFN,qBN)
qFN_check = QuatSum(qFB, qBN)
print("Quaternion qFN:", qFN)
print("Quaternion qBN:", qBN)   
print("Direct Method")
print("Quaternion qFB:", qFB)
print("Check qFN from qBN and qFB:", qFN_check)
DCM_FN = Quat2DCM(qFN)
DCM_BN = Quat2DCM(qBN)  
DCM_FB = DCM_FN @ DCM_BN.T
qFB_check = DCM2Quat(DCM_FB)
print("Indirect methos (DCM)")
print("Quaternion qFB from DCM:", qFB_check)

