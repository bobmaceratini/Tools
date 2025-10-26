import numpy as np
from RMKinematicTools import *

sBN = np.array([0.1,0.2,0.3])
sRB = np.array([-0.1,0.3,0.1])
DCM_BN = MRP2DCM(sBN)
DCM_RB = MRP2DCM(sRB)
DCM_RN = DCM_RB @ DCM_BN
sRN = DCM2MRP(DCM_RN)
ssum = MRPSum(sBN, sRB)
print("s1:", sBN)
print("s2:", sRB)    
print("s1 + s2 using MRPAdd:", ssum)
print("s1 + s2 using DCMs:", sRN)
print("Difference between MRPAdd and DCMs:", ssum - sRN)
print("")
print("-----------------")

sBN =  np.array([0.1,0.2,0.3])
sRN = np.array([0.5,0.3,0.1])
sBR = MRPSum(-sRN,sBN)
SBN_check = MRPSum(sRN, sBR)
print("sBR:", sBR)
print("Check sBN:", SBN_check)
sBN_DCM = MRP2DCM(sBN)
sRN_DCM = MRP2DCM(sRN)
sBR_DCM = sBN_DCM @ sRN_DCM.T
sBR_from_DCM = DCM2MRP(sBR_DCM)
sBN_DCM_check = DCM2MRP(sBR_DCM @ sRN_DCM)

print("sBR from DCMs:", sBR_from_DCM)   
print("Check sBN from DCMs:", sBN_DCM_check)