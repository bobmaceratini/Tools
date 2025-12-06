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

#-----------------------------------------------------------------------
Is1 = 86
Is2 = 85
Is3 = 113

Js = 0.13
Jt = 0.04
Jg = 0.03

IWs = 0.1

bigOmega_t0 = 14.4 # Initial RW speed
gamma_t0 = 0 # Initial Gimbal position
gamma_dot_t0 = 0 # Initial Gimbal speed

s_BN_t0 = np.array([0.1, 0.2, 0.3])  # Initial MRP attitude of Nano Spacecraft w.r.t. Inertial Frame
w_BN_B_t0 = np.array([0.01, -0.01, 0.005])  # Initial angular velocity of Nano Spacecraft w.r.t. Inertial Frame expressed in Body Frame in rad/s    

Is_v = np.array([Is1,Is2,Is3])  # Inertia Tensor elements
Ig_v = np.array([Js,Jt,Jg])  # Inertia Tensor elements

tetaG = 54.75/180*np.pi # Gimbal angular position
gg_B_t0 = np.array([np.cos(tetaG), 0, np.sin(tetaG)])
gs_B_t0 = np.array([0,1,0])
gt_B_t0 = np.cross(gg_B_t0, gs_B_t0)


L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   

tstep = 0.1
tmax = 30+tstep
time = np.arange(0, tmax, tstep)

sigma,omega,angles,gamma_dot,gamma,bigOmega,H_B,T = EOM_MRP_VSCMG_Single_Integrator(Is_v,Ig_v,IWs,s_BN_t0, w_BN_B_t0, time, 
                                                                           gs_B_t0, gt_B_t0, gg_B_t0, gamma_t0, 
                                                                           gamma_dot_t0, bigOmega_t0, L)

plt.figure(figsize=(10, 6))
plt.plot(time, sigma[:,0], label='s(1)', color='blue')
plt.plot(time, sigma[:,1], label='s(2)', color='green')
plt.plot(time, sigma[:,2], label='s(3)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('MRP components')
plt.title('Body MRP, BN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, omega[:,0], label='w(1)', color='blue')
plt.plot(time, omega[:,1], label='w(2)', color='green')
plt.plot(time, omega[:,2], label='w(3)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('omega')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, bigOmega[:,0], label='Omega', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('big Omega')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, gamma[:,0], label='gamma', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('rad')
plt.title('gamma')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()