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
mP = 150
r = 1
h = 3
L = 4
k = 1
d = 0.1
r_HB_B = np.array([-r,h/2])
IB2 = 500
IP2 = 200
teta_0 = 0

#--------------------------------------------------------------------------------------------
# Initial Conditions
teta_0 = 75./180.*np.pi
teta_dot_0 = 0
phi_0 = 25./180.*np.pi
phi_dot_0 = 0
xB_0 = 0
zB_0 = 0
xB_dot_0 = 0
zB_dot_0 = 0

#--------------------------------------------------------------------------------------------
# Derived Parameters
delta = np.arctan2(r_HB_B[1],r_HB_B[0])
r_HB = np.sqrt(np.dot(r_HB_B,r_HB_B))

#--------------------------------------------------------------------------------------------
# Auxiliary functions
# Funzione di aggiornamento dell'animazione
def update(frame):
    
    pts_body, pts_panel = get_shape(xB[frame], zB[frame], phi[frame], theta[frame])

    pts_body_closed = np.vstack([pts_body, pts_body[0]])
    pts_panel_closed = np.vstack([pts_panel, pts_panel[0]])

    pts_panel_closed = np.vstack([pts_panel])
    pts = np.vstack([pts_body_closed ,pts_panel_closed])

    rect_patch.set_data(pts[:, 0], pts[:, 1])
    return rect_patch,

# Funzione per ottenere i vertici del rettangolo ruotato
def get_shape(xB, zB, phy,theta):
    # Vertici rispetto al baricentro (non ruotati)
    pts_Body = np.array([
        [ h/2, -r],
        [ h/2,  r],
        [-h/2,  r],
        [-h/2,  -r]
    ])

    pts_Panel = np.array([
        [0, -d],
        [ L, -d],
        [ L,  d],
        [0,  d]
    ])

    R_body = np.array([
        [np.cos(phy), -np.sin(phy)],
        [np.sin(phy),  np.cos(phy)]
    ])

    R_panel = np.array([
        [np.cos(phy-np.pi/2+theta), -np.sin(phy-np.pi/2+theta)],
        [np.sin(phy-np.pi/2+theta),  np.cos(phy-np.pi/2+theta)]
    ])


    # Ruota e trasla i vertici
    pts_Body_rot = pts_Body @ R_body.T
    pts_Body_rot[:, 0] += zB
    pts_Body_rot[:, 1] += xB

    pts_Panel_rot = pts_Panel @ R_panel.T
    pts_Panel_rot[:, 0] += pts_Body_rot[0, 0]
    pts_Panel_rot[:, 1] += pts_Body_rot[0, 1]

    return pts_Body_rot, pts_Panel_rot

#--------------------------------------------------------------------------------------------
def MassMatrix(q,q_dot):
    xB, zB, phi, eta = q
    xB_dot, zB_dot, phi_dot, eta_dot = q_dot
    M = np.zeros([4,4])
    M[0,:] = mB+mP
    M[0,1] = 0
    M[0,2] = mP*r_HB*np.cos(phi-delta)
    M[0,3] = mP*L/2*np.sin(eta)

    M[1,0] = 0
    M[1,1] = mB+mP
    M[2,2] = -mP*r_HB*np.sin(phi-delta)
    M[1,3] = mP*L/2*np.sin(eta)

    M[2,0] = mP*r_HB*np.cos(phi-delta)
    M[2,1] = -mP*r_HB*np.sin(phi-delta)
    M[2,2] = IB2 + mP*r_HB**2
    M[2,3] = mP*L/2*np.sin(eta-phi+delta)

    M[3,0] = mP*L/2*np.sin(eta)
    M[3,1] = mP*L/2*np.cos(eta)
    M[3,2] = mP*L/2*np.sin(eta-phi+delta)
    M[3,3] = IP2 + mP/4*L**2
    
    return M

#--------------------------------------------------------------------------------------------
def GMatrix(q,q_dot):
    xB, zB, phi, eta = q
    xB_dot, zB_dot, phi_dot, eta_dot = q_dot
    G = np.zeros([4,1])
    G[0,0] = -mP*(phi_dot**2)*r_HB*np.sin(phi-delta) + mP*L/2*(eta_dot**2)*np.cos(eta)
    G[1,0] = -mP*(phi_dot**2)*r_HB*np.cos(phi-delta) - mP*L/2*(eta_dot**2)*np.sin(eta)
    G[2,0] = mP*L/2*(eta_dot**2)*r_HB*np.cos(eta-phi+delta) - k*(eta-phi)
    G[3,0] = -mP*L/2*(phi_dot**2)*r_HB*np.cos(eta-phi+delta) + k*(eta-phi)
    return G


#--------------------------------------------------------------------------------------------
def CalcQdotdot(q,q_dot):
    xB, zB, phi, eta = q
    xB_dot, zB_dot, phi_dot, eta_dot = q_dot

    M = MassMatrix(q,q_dot)
    G = GMatrix(q,q_dot)
    Mi = np.linalg.inv(M)

    F = -Mi @ G
    return F.reshape(4,)

#--------------------------------------------------------------------------------------------
def FunctionEval(q, q_dot):

    xB, zB, phi, eta = q
    xB_dot, zB_dot, phi_dot, eta_dot = q_dot

    F = CalcQdotdot(q,q_dot)
    q_dot = q_dot
    q_dot_dot = F

    return q_dot, q_dot_dot
    
#--------------------------------------------------------------------------------------------
def Integrator(q_0,_qdot_0,time, Ts):
    Np = len(time)
    q = np.zeros([4,Np])
    q_dot = np.zeros([4,Np])

    q[:,0] = q_0
    q_dot[:,0] = q_dot_0

    for index in range(1, Np):

        k1_q, k1_q_dot = FunctionEval(q[:,index-1] , q_dot[:,index-1] )
        k2_q, k2_q_dot = FunctionEval(q[:,index-1]+0.5*k1_q , q_dot[:,index-1]+0.5*k1_q_dot)
        k3_q, k3_q_dot = FunctionEval(q[:,index-1]+0.5*k2_q , q_dot[:,index-1]+0.5*k2_q_dot)
        k4_q, k4_q_dot = FunctionEval(q[:,index-1]+k3_q , q_dot[:,index-1]+k3_q_dot)

        k_q = (k1_q+2*k2_q+2*k3_q+k4_q)/6.0
        k_q_dot = (k1_q_dot+2*k2_q_dot+2*k3_q_dot+k4_q_dot )/6.0

        q[:,index] = q[:,index-1] + Ts* k_q
        q_dot[:,index] = q_dot[:,index-1] + Ts* k_q_dot.reshape(4,)

    return q, q_dot    

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# Main Program

# Simulation Configuration and array initialization
Ts = 0.01
t = np.arange(0,100,Ts)
Np = len(t)

q_0 = np.zeros([4])
q_dot_0 = np.zeros([4])

q_0[0] = xB_0
q_0[1] = zB_0
q_0[2] = phi_0
q_0[3] = teta_0 + phi_0
q_dot_0[0] = xB_dot_0
q_dot_0[1] = zB_dot_0
q_dot_0[2] = phi_dot_0
q_dot_0[3] = teta_dot_0 + phi_dot_0

q, q_dot= Integrator(q_0, q_dot_0, t, Ts)

xB = q[0,:]
zB = q[1,:]
phi = q[2,:]
theta = q[3,:] - phi

#x_t = 2
#z_t = 3 * np.sin(t)
#Phy_t = t  # rotazione lineare

# ---------------------------------------------------------
# Setup figura
fig, ax = plt.subplots()
ax.set_aspect('equal')
size = 8
ax.set_xlim(-size, size)
ax.set_ylim(-size, size)

# Patch iniziale del rettangolo
rect_patch, = ax.plot([], [], 'b-')

# ---------------------------------------------------------
# Animazione
ani = FuncAnimation(fig, update, range(0, len(t), 5), interval=0, blit=True)
plt.show()

# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, xB, label='xB', color='blue')
plt.plot(t, zB, label='zB', color='red')
plt.xlabel('Time [s]')
plt.title('Hub Center Of Mass Coordinates')
plt.ylabel('m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, phi/np.pi*180, label='phi', color='blue')
plt.plot(t, theta/np.pi*180, label='theta', color='red')
plt.xlabel('Time [s]')
plt.title('Hub and Panel Angles')
plt.ylabel('deg')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.show()
