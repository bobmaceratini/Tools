import numpy as np  
from QuaternionTools import *

eu= eu=np.array([20,10,-10]) * np.pi/180
DCM1 = ZYX2DCM(eu)
q1 = DCM2Quat(DCM1)
DCM2 = Quat2DCM(q1)
DCM3 = ZYX2Quat2(eu)
print("ZYX Euler angles (deg):", eu*180/np.pi)  
print("Direction Cosine Matrix from ZYX Euler angles:\n", DCM1)
print("Quaternion from DCM:\n", q1)
print("Reconstructed DCM from Quaternion:\n", DCM2)
print("Quaternion from ZYX Euler angles directly:\n", DCM3)
