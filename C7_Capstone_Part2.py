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
r_HB_B = np.array([-r,0.0,h/2])
IB2 = 500
IP2 = 200
teta_0 = 0

#--------------------------------------------------------------------------------------------
# Derived Parameters

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# Auxiliary functions
# Funzione di aggiornamento dell'animazione
def update(frame):
    xc = 2
    zc = 4
    phy = Phy_t[frame]

    pts_body, pts_panel = get_shape(xc, zc, phy, phy)

    pts_body_closed = np.vstack([pts_body, pts_body[0]])
    pts_panel_closed = np.vstack([pts_panel, pts_panel[0]])
#    pts_body_closed = np.vstack([pts_body])
    pts_panel_closed = np.vstack([pts_panel])
    pts = np.vstack([pts_body_closed ,pts_panel_closed])

    rect_patch.set_data(pts[:, 0], pts[:, 1])
    return rect_patch,


# ---------------------------------------------------------
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
# Main Program
# ---------------------------------------------------------
# Initial Conditions
teta_0 = 75./180.*np.pi
teta_dot_0 = 0
phi_0 = 25./180.*np.pi
phi_dot_0 = 0
xB_0 = 0
zB_0 = 0
xB_dot_0 = 0
zB_dot_0 = 0

# ---------------------------------------------------------
# Simulation Configuration
Ts = 0.01
t = np.arange(0,200,Ts)

Np = len(t)

q = np.zeros([8,Np])


x_t = 2
z_t = 3 * np.sin(t)
Phy_t = t  # rotazione lineare

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
# ---------------------------------------------------------
# Animazione
ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

plt.show()
