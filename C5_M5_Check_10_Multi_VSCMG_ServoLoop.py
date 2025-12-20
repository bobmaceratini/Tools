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

Is1 = 86.0
Is2 = 85.0
Is3 = 113.0

Js = 0.13
Jt = 0.04
Jg = 0.03

IWs = 0.1
num_gimb = 4

scale = 0.017453292519943295

W0 = 14.4
bigOmega_t0 = np.zeros(num_gimb) # Initial RW speeds for 4 VSCMGs
bigOmega_t0[0] = W0 # 
bigOmega_t0[1] = W0 # 
bigOmega_t0[2] = W0 # 
bigOmega_t0[3] = W0 # 

gamma_dot_t0 = np.zeros(num_gimb) # Initial Gimbal speed

gamma_t0 = np.zeros(num_gimb) # Initial Gimbal positions
gamma_t0[0] = 0/180*np.pi # 
gamma_t0[1] = 0/180*np.pi # 
gamma_t0[2] = 90/180*np.pi # 
gamma_t0[3] = -90/180*np.pi # 

gg_B_t0 = np.zeros((3,num_gimb))
gs_B_t0 = np.zeros((3,num_gimb))
gt_B_t0 = np.zeros((3,num_gimb))

tetaG = 54.75/180*np.pi # Gimbal angular position for VSCMG 

gg_B_t0[:,0] = np.array([np.cos(tetaG), 0, np.sin(tetaG)]  )  # Gimbal axis for VSCMG 0
gg_B_t0[:,1] = np.array([-np.cos(tetaG), 0, np.sin(tetaG)]  )  # Gimbal axis for VSCMG 1
gg_B_t0[:,2] = np.array([0, np.cos(tetaG), np.sin(tetaG)]  )  # Gimbal axis for VSCMG 2
gg_B_t0[:,3] = np.array([0, -np.cos(tetaG),  np.sin(tetaG)]  )  # Gimbal axis for VSCMG 3

gs_B_t0[:,0] = np.array([0,1,0])
gs_B_t0[:,1] = np.array([0,-1,0])
gs_B_t0[:,2] = np.array([1,0,0])
gs_B_t0[:,3] = np.array([-1,0,0])

gt_B_t0[:,0] = np.cross(gg_B_t0[:,0], gs_B_t0[:,0])
gt_B_t0[:,1] = np.cross(gg_B_t0[:,1], gs_B_t0[:,1]) 
gt_B_t0[:,2] = np.cross(gg_B_t0[:,2], gs_B_t0[:,2])
gt_B_t0[:,3] = np.cross(gg_B_t0[:,3], gs_B_t0[:,3])


s_BN_t0 = np.array([0.1, 0.2, 0.3])  # Initial MRP attitude of Nano Spacecraft w.r.t. Inertial Frame
w_BN_B_t0 = np.array([0.01, -0.01, 0.005]) # Initial angular velocity of Nano Spacecraft w.r.t. Inertial Frame expressed in Body Frame in rad/s    

Is_v = np.array([Is1,Is2,Is3])  # Space craft Inertia Tensor elements
Ig_v = np.array([Js,Jt,Jg])  # Gimbal Inertia Tensor elements

L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   

tstep = 0.1
tmax = 30+tstep*10
time = np.arange(0, tmax, tstep)
N = len(time)

bigOmega_dot_ref = np.zeros((num_gimb,len(time)))
gamma_dot_ref = np.zeros((num_gimb,len(time)))
sigma_ref = np.zeros((3,len(time)))
sigma_dot_ref = np.zeros((3,len(time)))
omega_ref = np.zeros((3,len(time)))

f1 = 0.02
f2 = 0.03
bigOmega_dot_ref[:,:] = np.array([np.sin(f1*time), np.cos(f1*time), -np.cos(f2*time), -np.cos(f2*time)])*scale
gamma_dot_ref[:,:] = np.array([np.sin(f1*time), np.cos(f1*time), -np.cos(f2*time), -np.cos(f2*time)])*scale                     
#bigOmega_dot_ref[:,:] = np.array([np.sin(f1*time)])*0
#gamma_dot_ref[:,:] = np.array([np.sin(f1*time)])                          


#gamma_dot_t0[0] = gamma_dot_ref[0,0] # 
#gamma_dot_t0[1] = gamma_dot_ref[1,0] #
#gamma_dot_t0[2] = gamma_dot_ref[2,0] #
#gamma_dot_t0[3] = gamma_dot_ref[3,0] #


sigma,omega,angles,gamma_dot,gamma,bigOmega,H_N,T,us,ug = EOM_MRP_VSCMG_Multi_CTRLIntegrator(num_gimb, Is_v,Ig_v,IWs,s_BN_t0, w_BN_B_t0, time, 
                                                                           gs_B_t0, gt_B_t0, gg_B_t0, gamma_t0, 
                                                                           gamma_dot_t0, bigOmega_t0, L,bigOmega_dot_ref, gamma_dot_ref)

bigOmega_dot = np.zeros((num_gimb, N))
for i in range(num_gimb):
    bigOmega_dot[i,:] = np.gradient(bigOmega[i,:], tstep)

t_eval = 30
index_t_eval = np.argmin(np.abs(time - t_eval))

H_N_t = H_N[:,index_t_eval]
T_t = T[index_t_eval]
sigma_t = sigma[:,index_t_eval]
omega_t = omega[:,index_t_eval]
gamma_t = gamma[:,index_t_eval]
gamma_dot_t = gamma_dot[:,index_t_eval]
bigOmega_t = bigOmega[:,index_t_eval] 


print("At time t =", t_eval, "s:")
print("\tH_N = [{:.4f},{:.4f},{:.4f}]".format(H_N_t[0], H_N_t[1], H_N_t[2]))    
print("\tT = {:.4f}".format(T_t))
print("\tsigma_BN = [{:.4f},{:.4f},{:.4f}]".format(sigma_t[0], sigma_t[1], sigma_t[2]))
print("\tomega_BN_B=[{:.5f},{:.5f},{:.5f}]".format(omega_t[0], omega_t[1], omega_t[2]))
print("\tOmega =[{:.4f},{:.4f},{:.4f},{:.4f}]".format(bigOmega_t[0], bigOmega_t[1], bigOmega_t[2], bigOmega_t[3]))
print("\tgamma = [{:.4f},{:.4f},{:.4f},{:.4f}]".format(gamma_t[0], gamma_t[1], gamma_t[2], gamma_t[3]))


plt.figure(figsize=(10, 6))
plt.plot(time, sigma[0,:], label='s(1)', color='blue')
plt.plot(time, sigma[1,:], label='s(2)', color='green')
plt.plot(time, sigma[2,:], label='s(3)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('MRP components')
plt.title('Body MRP, BN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, omega[0,:], label='w(1)', color='blue')
plt.plot(time, omega[1,:], label='w(2)', color='green')
plt.plot(time, omega[2,:], label='w(3)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('omega')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, bigOmega[i,:], label=f'BigOmega{i}')

plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('big Omega')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, gamma[i,:], label=f'gamma{i}')
plt.xlabel('Time [s]')
plt.ylabel('rad')
plt.title('gamma')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, bigOmega_dot[i,:], label=f'bigOmega_dot{i}')
    plt.plot(time, bigOmega_dot_ref[i,:], '--',label=f'bigOmega_dot_ref{i}')
plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('bigOmega dot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, gamma_dot[i,:], label=f'gamma_dot{i}')
    plt.plot(time, gamma_dot_ref[i,:], '--',label=f'gamma_dot_ref{i}')
plt.xlabel('Time [s]')
plt.ylabel('rad/s')
plt.title('gamma dot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, us[i,:], label=f'us{i}')
plt.xlabel('Time [s]')
plt.ylabel('Nm')
plt.title('torque inputs Us')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_gimb):
    plt.plot(time, ug[i,:], label=f'ug{i}')
plt.xlabel('Time [s]')
plt.ylabel('Nm')
plt.title('torque inputs Ug')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, H_N[0,:], label='H1')
plt.plot(time, H_N[1,:], label='H2')
plt.plot(time, H_N[2,:], label='H3')
plt.xlabel('Time [s]')
plt.ylabel('Nm')
plt.title('H')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, T, label='T', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('J,')
plt.title('T')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

