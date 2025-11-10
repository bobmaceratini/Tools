from RMKinematicTools import *
from RMKineticsTools import *
from RMAttitudeStaticDeterminationTools import *


IC_B = np.array([[10,1,-1],[1,5,1],[-1,1,8]])
print("Intertia Tensor vs COM, B frame: ")
print(IC_B)

wB = np.array([0.01,-0.01,0.01])
T = 0.5 *wB.T @ IC_B @ wB
print("Rotational kinetic energy [J]:")
print(T)