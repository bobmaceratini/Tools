import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *
from RMVSCMG_RW import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from RMKinematicTools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#--------------------------------------------------------------------------------------------
# Input data
mB = 500
mP = 50
L = 2
w = 1
h = 3
r = 1
r_CN_0 = np.array([10.0,20.0,30.0])
v_CN_0 = np.array([0.01, 0.01, 0.02])
r_HB = np.array([0.5,0.0,1.5])
sigma_BN_0 = np.array([0.1,0.2,0.3])
w_BN_0 = np.array([1,-1,1])/180.0*np.pi
teta_0 = 25.0/180.0*np.pi
teta_dot_0 = 12.0/180.0*np.pi

#--------------------------------------------------------------------------------------------
# Derived Parameters
r_PH_P = np.array([L/2,0,0])
IB1 = 1/12*mB*(3*r**2 + h**2)
IB2 = IB1
IB3 = 1/2*mB*r**2
IB = np.diag([IB1, IB2, IB3])
IP1 = 1/12*mP*(w**2)
IP2 = 1/12*mP*(L**2)
IP3 = 1/12*mP*(L**2 + w**2)
IP = np.diag([IP1, IP2, IP3])

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# Auxiliary functions

# Calculation of PH Matrix
def Calc_PH(teta):
    return np.array([[np.cos(teta),0,-np.sin(teta)],[0,1,0],[np.sin(teta),0,np.cos(teta)]])

# Calculation of Body and Panel Center of Masses Position
def Calc_R_BP_N(teta,sigma_BN,r_CN_N):
    BN = MRP2DCM(sigma_BN)
    PH = Calc_PH(teta)
    r_BN_N = r_CN_N - mP/(mP+mB)*BN.T @ (r_HB + HB.T @ PH.T @ r_PH_P)
    r_PN_N =  r_BN_N + BN.T @ (HB.T @ PH.T @ r_PH_P + r_HB)
    return r_BN_N, r_PN_N

# Calculation of Body and Panel Center of Masses Inertial Velocities
def Calc_V_BP_N(teta, sigma_BN, v_CN, omega_BN_B, teta_dot):
    omega_PN_B = np.array([omega_BN_B[0],omega_BN_B[1]+teta_dot,omega_BN_B[2]])
    BN = MRP2DCM(sigma_BN)
    PH = Calc_PH(teta)
    r_HB_B = r_HB
    r_PH_B = HB.T @ PH.T @ r_PH_P 
    v_BN_N = v_CN - mP/(mP+mB)*BN.T@(np.cross(omega_PN_B,r_PH_B)+np.cross(omega_BN_B,r_HB_B))
    v_PN_N = BN.T@(np.cross(omega_PN_B,r_PH_B)+np.cross(omega_BN_B,r_HB_B))+v_BN_N
    return v_BN_N, v_PN_N
                        
# Calculation of Spacecarft Kientic Energy
def Calc_Kinetic_Energy(teta, sigma_BN, r_CN, v_CN, omega_BN, teta_dot):

    v_BN_N, v_PN_N = Calc_V_BP_N(teta, sigma_BN, v_CN, omega_BN, teta_dot)

    omega_PN_B = np.array([omega_BN[0],omega_BN[1]+teta_dot,omega_BN[2]])

    PH = Calc_PH(teta)
    PB = PH @ HB
    omega_PN_P = PB @ omega_PN_B

    T_trans = 1/2*mB* np.dot(v_BN_N,v_BN_N) + 1/2*mP* np.dot(v_PN_N,v_PN_N)
    T_rot = 1/2*omega_BN @ IB @ omega_BN + 1/2*omega_PN_P @ IP @ omega_PN_P
    T = T_trans + T_rot

    return  T


#--------------------------------------------------------------------------------------------
# Task 1.1
HB = np.array([[-1,0,0,],[0,1,0],[0,0,-1]])
print("HB = \n", HB)

#--------------------------------------------------------------------------------------------
# Task 1.2
P0H = Calc_PH(teta_0)
print("P0H = \n", P0H)

#--------------------------------------------------------------------------------------------
# Task 1.3
B0N = MRP2DCM(sigma_BN_0)
P0N = P0H @ HB @ B0N
print("P0N = \n", P0N)

#--------------------------------------------------------------------------------------------
# Task 2.1
r_BN, r_PN = Calc_R_BP_N(teta_0, sigma_BN_0, r_CN_0)
print("r_BN_N = ", r_BN)

#--------------------------------------------------------------------------------------------
# Task 2.2
print("r_PN_N = ", r_PN)

#--------------------------------------------------------------------------------------------
# Task 2.3
Left_side = mB*r_BN + mP*r_PN
Right_Side = (mB+mP)*r_CN_0
print("mB*r_BN_0 + mP*r_PN_0 = ", Left_side)
print("(mB+mP)*r_CN_0 = ", Right_Side)

#--------------------------------------------------------------------------------------------
# Task 3.1
v_BN_0, v_PN_0 = Calc_V_BP_N(teta_0, sigma_BN_0, v_CN_0, w_BN_0, teta_dot_0)
print("v_BN_0 = ", v_BN_0)

#--------------------------------------------------------------------------------------------
# Task 3.1
print("v_PN_0 = ", v_PN_0)

#--------------------------------------------------------------------------------------------
# Task 3.3
Left_side = mB*v_BN_0 + mP*v_PN_0
Right_Side = (mB+mP)*v_CN_0
print("mB*v_BN_0 + mP*v_PN_0 = ", Left_side)
print("(mB+mP)*v_CN_0 = ", Right_Side)

#--------------------------------------------------------------------------------------------
# Task 3.4
T = Calc_Kinetic_Energy(teta_0, sigma_BN_0, r_CN_0, v_CN_0, w_BN_0, teta_dot_0)
print("Kinetic Energy: ", T)
