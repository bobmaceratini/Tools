import numpy as np
from RMKinematicTools import *

s=np.array([0.1,0.2,0.3])
DCM = MRP2DCM(s)
sback = DCM2MRP(DCM)

print("s:", s)
print("DCM:\n", DCM)
print("s back from DCM:", sback)

DCM2 = np.array([[0.763314,0.0946746,-0.639053],
                 [-0.568047,-0.372781,-0.733728],
                 [-0.307692,0.923077,-0.230769]])

s2 = DCM2MRP(DCM2)
shadow = MRP2Shadow(s2)
DCM2_back = MRP2DCM(shadow)
print("\nGiven DCM:\n", DCM2)
print("s from given DCM:", s2)
print("Shadow of s:", shadow)
print("DCM back from s:\n", DCM2_back)
