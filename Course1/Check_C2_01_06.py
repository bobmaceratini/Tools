from RMKinematicTools import *
from RMKineticsTools import *
from RMAttitudeStaticDeterminationTools import *

m = 12.5
I_B = np.array([[10,1,-1],[1,5,1],[-1,1,8]])
print("Intertia Tensor vs COM, B frame: ")
print(I_B)

teta_eu321 = np.array([-10,10,5])/180*np.pi
r_CP_Nf = np.array([-0.5,0.5,0.25])

# method 1
BN = Eu2DCM_ZYX(teta_eu321)

r_CP_Bf = BN @ r_CP_Nf

R_CP_Bf_tilde = tilde(r_CP_Bf)

IB_P = I_B + m * (R_CP_Bf_tilde @ R_CP_Bf_tilde.T)
print("Method1: IB_P")
print(IB_P)

#method 2
IN_C = BN.T @ I_B @ BN

R_CP_Nf_tilde = tilde(r_CP_Nf)

IN_P = IN_C + m * R_CP_Nf_tilde @ R_CP_Nf_tilde.T

IB_P2 = BN @ IN_P @ BN.T
print("Method2: IB_P")
print(IB_P2)
