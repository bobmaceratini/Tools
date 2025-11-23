import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from RMKinematicTools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

rLMO = 3796.19  # Radius of Nano Spacecraft Orbit in km
OmLMO = 20.0/180*np.pi  # Longitude of Ascending Node of Nano Spacecraft Orbit in rad
iLMO = 30.0/180*np.pi  # Inclination of Nano Space craft Orbit in rad
wLMO = 0.000884797  # Argument of Periapsis of Nano Spacecraft Orbit in rad/s
tetaLMO_0 = 60.0/180*np.pi  # True Anomaly of Nano Spacecraft at t=0 in rad

rGMO = 20424.2   # Radius of Mother Spacecraft Orbit in km
OmGMO = 0.0/180*np.pi  # Longitude of Ascending Node of Mother Spacecraft Orbit in rad
iGMO = 0.0/180*np.pi  # Inclination of Mother Space craft Orbit in rad
wGMO = 0.0000709003   # Argument of Periapsis of Mother Spacecraft Orbit in rad/s
tetaGMO_0 = 250.0/180*np.pi  # True Anomaly of Mother Spacecraft at t=0 in rad

s_BN_t0 = np.array([0.3, -0.4, 0.5])  # Initial MRP attitude of Nano Spacecraft w.r.t. Inertial Frame
w_BN_B_t0 = np.array([1.0, 1.75, -2.20])/180*np.pi  # Initial angular velocity of Nano Spacecraft w.r.t. Inertial Frame expressed in Body Frame in rad/s    
I1 = 10.0
I2 = 5.0
I3 = 7.5
InertiaTensor = np.array([[I1,0,0],[0,I2,0],[0,0,I3]])  # Inertia Tensor of Nano Spacecraft in kg*m^2

# Mod 2 Functions
def get_DCM(Om, i, w, time, teta_0):
    teta = w*time + teta_0
    ZXZ = np.array([Om, i , teta])
    DCM = Eu2DCM_ZXZ(ZXZ)
    return DCM

def get_position(r, Om, i, w, time, teta_0):
    DCM = get_DCM(Om, i, w, time, teta_0)
    rH = np.array([r,0,0])
    rN = DCM.T @ rH
    return rN

def get_velocity(r, Om, i, w, time, teta_0):
    DCM = get_DCM(Om, i, w, time, teta_0)
    vH = np.array([0,r*w,0])
    vN = DCM.T @ vH
    return vN

def get_HN(time):
    HN = get_DCM(OmLMO, iLMO, wLMO, time, tetaLMO_0)
    return HN
    
def get_RsN(time):
    RsN = np.array([[-1,0,0],[0,0,1],[0,1,0]])
    return RsN

def get_RnH(time):
    RnH = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    return RnH

def get_wRsN(time):
    w = np.array([0,0,0])
    return w

def get_wHN(time):
    wHN_H = np.array([0,0,wLMO])
    wHN_N = get_HN(time).T @ wHN_H
    return wHN_N

def get_RnN(time):
    HN = get_HN(time)
    RnH = get_RnH(time)
    RnN = RnH @ HN
    return RnN

def get_wRnN(time):
    wRnN = get_wHN(time)
    return wRnN

def get_rLMO(time):
    rLMO_t = get_position(rLMO, OmLMO, iLMO, wLMO, time, tetaLMO_0)
    return rLMO_t

def get_rGMO(time):
    rGMO_t = get_position(rGMO, OmGMO, iGMO, wGMO, time, tetaGMO_0)
    return rGMO_t   

def get_Delta_rMO(time):
    rLMO_t = get_rLMO(time)
    rGMO_t = get_rGMO(time)
    Delta_rMO = rGMO_t - rLMO_t
    return Delta_rMO

def get_RcN(time):
    Delta_r = get_Delta_rMO(time)
    rc1 = -Delta_r / np.linalg.norm(Delta_r)
    rc2 = np.cross(Delta_r, np.array([0,0,1]))
    rc2 = rc2 / np.linalg.norm(rc2)
    rc3 = np.cross(rc1, rc2)   
    rc3 = rc3 / np.linalg.norm(rc3) 
    RcN = np.column_stack((rc1, rc2, rc3))
    RcN = RcN.T
    return RcN

def get_wRcN(time,Delta_t=0.1):
    R_k = get_RcN(time)
    R_k1 = get_RcN(time + Delta_t)
    R_k1_k = R_k1 @ R_k.T
    s = DCM2MRP(R_k1_k)
    wRcN_B = (4/Delta_t) * s  
    wRcN_N = R_k.T @ wRcN_B
    return wRcN_N
    
def get_wRcN2(time,Delta_t):
    R_k = get_RcN(time)
    R_k1 = get_RcN(time + Delta_t)
    sk = DCM2MRP(R_k)
    sk1 = DCM2MRP(R_k1)
    sigdot = (sk1 - sk) / Delta_t
    wRcN_B = MRP_InvDifferential(sk,sigdot)
    wRcN_N = R_k.T @ wRcN_B
    return wRcN_N
    
def getTrackingError(s_BN, w_BN_B, DCM_RN, w_RN_N):
    s_RN = DCM2MRP(DCM_RN)
    s_BR = MRPSum(-s_RN,s_BN)
    if np.linalg.norm(s_BR) > 1:
        s_BR = MRP2Shadow(s_BR)
    DCM_BN = MRP2DCM(s_BN)
    w_BR_B = w_BN_B - DCM_BN @ w_RN_N
    return s_BR, w_BR_B


def printfile(filename,data):
    with open(filename, "w") as f:
    # Flatten della matrice e scrittura su una riga
        f.write(" ".join(f"{x:.7f}" for x in data.flatten()))
    return

def Animation(skip):
    # Vertici del cubo
    cube_vertices = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ])
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    # Setup figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale = 4
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])
    ax.set_zlim([-scale, scale])
    lines = [ax.plot([], [], [], 'b')[0] for _ in edges]
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_zlim([-scale, scale])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        psi, theta, phi = angles[frame]
        rot = R.from_euler('ZYX', [psi, theta, phi])
        rotated_vertices = rot.apply(cube_vertices)

        # Disegna il cubo
        #for start, end in edges:
        #    xs, ys, zs = zip(rotated_vertices[start], rotated_vertices[end])
        #    ax.plot(xs, ys, zs, 'b')

        # Terna solidale (assi del corpo)
        origin = np.array([0, 0, 0])+ get_rLMO(time[frame])/1000
        body_axes = np.eye(3)  # X, Y, Z unitari
        rotated_axes = rot.apply(body_axes)

        colors = ['r', 'b', 'g']  
        labels = ['X_body', 'Y_body', 'Z_body']

        for vec, color, label in zip(rotated_axes, colors, labels):
            ax.quiver(*origin, *vec, color=color, length=1.5, normalize=True)
            ax.text(*(vec * 1.2), label, color=color)
            ax.plot([0 ,origin[0]],
            [0, origin[1]],
            [0, origin[2]],
            color='orange', linewidth=2)

        ax.set_title(f"t = {time[frame]:.1f}s")

        return lines

    ani = FuncAnimation(fig, update, frames=range(0, len(time), skip), interval=10, blit=False)
    plt.show()
    return
    
def PlotSimulation(okplot,step=1):
    if okplot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, sigma[:,0], label='s(1)', color='blue')
        plt.plot(time, sigma_ref[:,0], '--', label='sref(1)', color='blue')
        plt.plot(time, sigma[:,1], label='s(2)', color='green')
        plt.plot(time, sigma_ref[:,1], '--', label='sref(2)', color='green')
        plt.plot(time, sigma[:,2], label='s(3)', color='orange')
        plt.plot(time, sigma_ref[:,2], '--', label='sref(3)', color='orange')
        plt.plot(time, snorm, label='s', color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('MRP components [m]')
        plt.title('Body MRP, BN')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(time, omega[:,0], label='s(1)', color='blue')
        plt.plot(time, omega_ref[:,0], '--', label='s_ref(1)', color='blue')
        plt.plot(time, omega[:,1], label='s(2)', color='green')
        plt.plot(time, omega_ref[:,1], '--', label='s_ref(2)', color='green')
        plt.plot(time, omega[:,2], label='s(3)', color='orange')
        plt.plot(time, omega_ref[:,2], '--', label='s_ref(3)', color='orange')
        plt.xlabel('Time [s]')
        plt.ylabel('angular velocity [rad/s]')
        plt.title('Angular velocity, BN')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(time, u[:,0], label='u(1)', color='blue')
        plt.plot(time, u[:,1], label='u(2)', color='green')
        plt.plot(time, u[:,2], label='u(3)', color='orange')
        plt.xlabel('Time [s]')
        plt.ylabel('Thruster Force [N]')
        plt.title('Trhuster commands')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        Animation(step)
    return

# Task 1
t = 450  # time in seconds for Nano Spacecraft
PLMO = get_position(rLMO, OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")
print("Task 1.1 - Position of Nano Spacecraft at t=450s (km): ", PLMO)
printfile("Task1p1.txt",PLMO)

VLMO = get_velocity(rLMO, OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")
print("Task 1.2 - Velocity of Nano Spacecraft at t=450s (km/s): ", VLMO)
printfile("Task1p2.txt",VLMO)

t = 1150  # time in seconds for Mother Spacecraft
PGMO = get_position(rGMO, OmGMO, iGMO, wGMO, t, tetaGMO_0)
print("---------------------------------------------------")
print("Task 1.3 - Position of Mother Spacecraft at t=1150s (km): ", PGMO)
printfile("Task1p3.txt",PGMO)

VGMO = get_velocity(rGMO, OmGMO, iGMO, wGMO, t, tetaGMO_0)
print("---------------------------------------------------")
print("Task 1.4 - Velocity of Mother Spacecraft at t=1150s (km/s): ", VGMO)
printfile("Task1p4.txt",VGMO)

# Task 2
t = 300  # time in seconds
DCM = get_DCM(OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")    
print("Mod 2 Task 2 - DCM of Nano Spacecraft at t=300s: \n", DCM)
printfile("Task2p1.txt",DCM)

# Task 3

# Task 4    
t = 330  # time in seconds
RnN = get_RnN(t)
print("---------------------------------------------------")
print("Task 4 - Matrix RnN at t=330s : \n", RnN)
printfile("Task4p1.txt",RnN)

wRnN = get_wRnN(t)
print("---------------------------------------------------")
print("Task 4 - RnN angular velocity at t=330s (rad/s): \n", wRnN)
printfile("Task4p2.txt",wRnN)

# Task 5
t = 330  # time in seconds
RcN = get_RcN(t)
print("---------------------------------------------------")
print(f"Task 5 - Matrix RcN at t={t}s: \n", RcN)
printfile("Task5p1.txt",RcN)

wRcN = get_wRcN(t,0.01)
print("---------------------------------------------------")
print(f"Task 5 - RcN angular velocity at t={t}s (rad/s): \n", wRcN)
printfile("Task5p2.txt",wRcN)

#Task 6
t = 0  # time in seconds
RsN = get_RsN(t)
wRsN = get_wRsN(t)
s_BRs, w_BR_Bs = getTrackingError(s_BN_t0, w_BN_B_t0, RsN, wRsN)
print("---------------------------------------------------")
print("Task 6 - Tracking error w.r.t. Sun frame Rs at t=0")
print("MRP attitude error s_BR: ", s_BRs)
print("Angular velocity error w_BR_B (rad/s): ", w_BR_Bs)
printfile("Task6p1.txt",s_BRs)
printfile("Task6p2.txt",w_BR_Bs)

RnN = get_RnN(t)
wRnN = get_wRnN(t)
s_BRn, w_BR_Bn = getTrackingError(s_BN_t0, w_BN_B_t0, RnN, wRnN)
print("---------------------------------------------------")
print("Task 6 - Tracking error w.r.t. Mars frame Rn at t=0")
print("MRP attitude error s_BR: ", s_BRn)
print("Angular velocity error w_BR_B (rad/s): ", w_BR_Bn)
printfile("Task6p3.txt",s_BRn)
printfile("Task6p4.txt",w_BR_Bn)

RcN = get_RcN(t)
wRcN = get_wRcN(t)
s_BRc, w_BR_Bc = getTrackingError(s_BN_t0, w_BN_B_t0, RcN, wRcN)
print("---------------------------------------------------")
print("Task 6 - Tracking error w.r.t. GMO frame Rc at t=0")
print("MRP attitude error s_BR: ", s_BRc)
print("Angular velocity error w_BR_B (rad/s): ", w_BR_Bc)
printfile("Task6p5.txt",s_BRc)
printfile("Task6p6.txt",w_BR_Bc)

#Task 7 (7.1, 7.2, 7.3, 7.4)
tstep = 1
tmax = 600+tstep
time = np.arange(0, tmax, tstep)    

sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time)
okplot = False
if okplot:
    plt.figure(figsize=(10, 6))
    plt.plot(time, omega[:,0]/np.pi*180, label='wBN_B(1)', color='blue')
    plt.plot(time, omega[:,1]/np.pi*180, label='wBN_B(2)', color='green')
    plt.plot(time, omega[:,2]/np.pi*180, label='wBN_B(3)', color='orange')

    plt.xlabel('Time [s]')
    plt.ylabel('Angular velocity [deg/s]')
    plt.title('Body Angular Velocity, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, sigma[:,0], label='sBN_(1)', color='blue')
    plt.plot(time, sigma[:,1], label='sBN_(2)', color='green')
    plt.plot(time, sigma[:,2], label='sBN_(3)', color='orange')

    plt.xlabel('Time [s]')
    plt.ylabel('MRP components [m]')
    plt.title('Body MRP, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, H[:,0], label='H_(1)', color='blue')
    plt.plot(time, H[:,1], label='H_(2)', color='green')
    plt.plot(time, H[:,2], label='H_(3)', color='orange')

    plt.xlabel('Time [s]')
    plt.ylabel('Angular Momentum [kgm2/s]')
    plt.title('Angular Momentum Components, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, T, label='Trot', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Kinetic Energy [J]')
    plt.title('Rotational Kinetic Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

t_eval = 500
index_t_eval = np.argmin(np.abs(time - t_eval))
omega_t = omega[index_t_eval,:]
H_t = InertiaTensor @ omega_t
T_t = 1/2 * omega_t.T @ InertiaTensor @ omega_t
s_t = sigma[index_t_eval,:]
H_t_N = MRP2DCM(s_t).T @ H_t


print("---------------------------------------------------")
print(f"Task 7.1 - Angular Momentum at t={t_eval}s (kg*m^2/s): ", H_t)
printfile("Task7p1.txt",H_t)
print("---------------------------------------------------")
print(f"Task 7.2 - Rotational Kinetic energy at t={t_eval}s (J): ", T_t)
printfile("Task7p2.txt",T_t)
print("---------------------------------------------------")
print(f"Task 7.3 - MRP at t={t_eval}s (J): ", s_t)
printfile("Task7p3.txt",s_t)
print("---------------------------------------------------")
print(f"Task 7.4 - Angual Momentun at t={t_eval}s in N frame(J): ", H_t_N)
printfile("Task7p4.txt",H_t_N)

#Task 7 (7.5)
tstep = 1
tmax = 600+tstep
time = np.arange(0, tmax, tstep)    

uf = np.array([0.01, -0.01, 0.02])  # constant disturbance torque in N*m
sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time,uf)

okplot = False
if okplot:
    plt.figure(figsize=(10, 6))
    plt.plot(time, omega[:,0]/np.pi*180, label='wBN_B(1)', color='blue')
    plt.plot(time, omega[:,1]/np.pi*180, label='wBN_B(2)', color='green')
    plt.plot(time, omega[:,2]/np.pi*180, label='wBN_B(3)', color='orange')

    plt.xlabel('Time [s]')
    plt.ylabel('Angular velocity [deg/s]')
    plt.title('Body Angular Velocity, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, sigma[:,0], label='sBN_(1)', color='blue')
    plt.plot(time, sigma[:,1], label='sBN_(2)', color='green')
    plt.plot(time, sigma[:,2], label='sBN_(3)', color='orange')


    plt.xlabel('Time [s]')
    plt.ylabel('MRP components [m]')
    plt.title('Body MRP, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, H[:,0], label='H_(1)', color='blue')
    plt.plot(time, H[:,1], label='H_(2)', color='green')
    plt.plot(time, H[:,2], label='H_(3)', color='orange')

    plt.xlabel('Time [s]')
    plt.ylabel('Angular Momentum [kgm2/s]')
    plt.title('Angular Momentum Components, Body Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, T, label='Trot', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Kinetic Energy [J]')
    plt.title('Rotational Kinetic Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

t_eval = 100
index_t_eval = np.argmin(np.abs(time - t_eval))
s_t = sigma[index_t_eval,:]
print(f"Task 7.5 - MRP at t={t_eval}s (J): ", s_t)
printfile("Task7p5.txt",s_t)

#Task 8 
Tdecay = 120
Imax = np.max([I1,I2,I3])
Pg = 2*Imax/Tdecay
P = np.array([Pg,Pg,Pg])
Imin = np.min([I1,I2,I3])
K = Pg**2/Imin
Tdecay1 = 2*I1/Pg
Tdecay2 = 2*I2/Pg
Tdecay3 = 2*I3/Pg
damp1 = Pg/(np.sqrt(K*I1))
damp2 = Pg/(np.sqrt(K*I2))
damp3 = Pg/(np.sqrt(K*I3))

#Task 8.1
print("---------------------------------------------------")
print(f"Task 8.1 - K and P gains used in Task 8: P = {Pg} N, K = {K} Nms/rad")
print("---------------------------------------------------")
print("Verification of decay times (s): ", Tdecay1, Tdecay2, Tdecay3)
print("Verification of damping ratios: ", damp1, damp2, damp3)
gains = np.array([Pg, K])
printfile("Task8p1.txt",gains)

tstep = 1
tmax = 600+tstep
time = np.arange(0, tmax, tstep)
sigma_ref_S = np.zeros((len(time), 3))
omega_ref_S = np.zeros((len(time), 3))
for i in range(0,len(time)):
    t = time[i]
    RsN = get_RsN(t)
    sigma_ref_S[i]= DCM2MRP(RsN)
    if np.linalg.norm(sigma_ref_S[i])>1:
        sigma_ref_S[i] = MRP2Shadow(sigma_ref_S[i])
    omega_ref_S[i] = RsN.T @ get_wRsN(t)

sigma_ref =  sigma_ref_S
omega_ref = omega_ref_S

L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   
sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time, L,
                                                              K, P, 0, 0, 0, 1e100, sigma_ref_S, omega_ref_S)
snorm = np.zeros((len(time), 1))
for i in range(1, len(time)):
    snorm[i] = np.linalg.norm(sigma[i])

t_eval = 15
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 8.2 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task8p2.txt",sigma_teval)

t_eval = 100
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 8.3 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task8p3.txt",sigma_teval)

t_eval = 200
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 8.4 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task8p4.txt",sigma_teval)

t_eval = 400
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 8.5 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task8p5.txt",sigma_teval)

PlotSimulation(False)

#Task 9
tstep = 1
tmax = 7097*2+tstep
time = np.arange(0, tmax, tstep)
sigma_ref_N = np.zeros((len(time), 3))
omega_ref_N = np.zeros((len(time), 3))
for i in range(0,len(time)):
    t = time[i]
    RnN = get_RnN(t)
    sigma_ref_N[i]= DCM2MRP(RnN)
    if np.linalg.norm(sigma_ref_N[i])>1:
        sigma_ref_N[i] = MRP2Shadow(sigma_ref_N[i])
    omega_ref_N[i] = RnN.T @ get_wRnN(t)

sigma_ref =  sigma_ref_N
omega_ref = omega_ref_N

L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   
sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time, L,
                                                              K, P, 0, 0, 0, 1e100, sigma_ref, omega_ref)
snorm = np.zeros((len(time), 1))
for i in range(1, len(time)):
    snorm[i] = np.linalg.norm(sigma[i])

t_eval = 15
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 9.1 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task9p1.txt",sigma_teval)

t_eval = 100
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 9.2 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task9p2.txt",sigma_teval)

t_eval = 200
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 9.3 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task9p3.txt",sigma_teval)

t_eval = 400
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 9.4 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task9p4.txt",sigma_teval)

PlotSimulation(True,30)

#Task 10
tstep = 1
tmax = 7097*2+tstep
time = np.arange(0, tmax, tstep)
sigma_ref_C = np.zeros((len(time), 3))
omega_ref_C = np.zeros((len(time), 3))
for i in range(0,len(time)):
    t = time[i]
    RcN = get_RcN(t)
    sigma_ref_C[i]= DCM2MRP(RcN)
    if np.linalg.norm(sigma_ref_C[i])>1:
        sigma_ref_C[i] = MRP2Shadow(sigma_ref_C[i])
    omega_ref_C[i] = RcN.T @ get_wRcN(t)

sigma_ref =  sigma_ref_C
omega_ref = omega_ref_C

L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   
sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time, L,
                                                              K, P, 0, 0, 0, 1e100, sigma_ref, omega_ref)
snorm = np.zeros((len(time), 1))
for i in range(1, len(time)):
    snorm[i] = np.linalg.norm(sigma[i])

t_eval = 15
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 10.1 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task10p1.txt",sigma_teval)

t_eval = 100
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 10.2 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task10p2.txt",sigma_teval)

t_eval = 200
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 10.3 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task10p3.txt",sigma_teval)

t_eval = 400
index_t_eval = np.argmin(np.abs(time - t_eval))
sigma_teval = sigma[index_t_eval]
print("---------------------------------------------------")
print(f"Task 10.4 sigma @ t = {t_eval}, {sigma_teval}")
printfile("Task10p4.txt",sigma_teval)

PlotSimulation(True,30)
