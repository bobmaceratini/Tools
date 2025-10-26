import numpy as np  
from RMKinematicTools import *

print("")
print("----- Assignment 1 -----")
eu_312 = np.array([10,20,30]) * np.pi/180
DCM = Eu2DCM_ZYX(eu_312)
eu_313 = DCM2Eu_ZXZ(DCM)
DCM313 = Eu2DCM_ZXZ(eu_313)
print("312 Euler angles (deg):", eu_312*180/np.pi)  
print("Reconstructed 313 Euler angles from DCM (deg):", eu_313*180/np.pi)
print("Direction Cosine Matrix from 312 Euler angles:\n", DCM)
print("Reconstructed DCM from 313 Euler angles:\n", DCM313)

print("")

print("----- Assignment 2 -----")
eu_BN = np.array([10,20,30]) * np.pi/180
eu_RN = np.array([-5,5,5]) * np.pi/180
DCM_BN = Eu2DCM_ZYX(eu_BN)
DCM_RN = Eu2DCM_ZYX(eu_RN)  
DCM_BR = DCM_BN @ DCM_RN.T
eu_BR = DCM2Eu_ZYX(DCM_BR)  
print("Euler angles B-N (deg):", eu_BN*180/np.pi)  
print("Euler angles R-N (deg):", eu_RN*180/np.pi)       
print("Euler angles B-R from DCM (deg):", eu_BR*180/np.pi)
