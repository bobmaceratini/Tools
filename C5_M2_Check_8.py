import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from RMKinematicTools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
        origin = np.array([0, 0, 0])
        body_axes = np.eye(3)  # X, Y, Z unitari
        rotated_axes = rot.apply(body_axes)
        colors = ['r', 'b', 'g']  
        labels = ['X_body', 'Y_body', 'Z_body']

        for vec, color, label in zip(rotated_axes, colors, labels):
            ax.quiver(*origin, *vec, color=color, length=1.5, normalize=True)
            ax.text(*(vec * 1.2), label, color=color)
            ax.plot([0 ,origin[0]], [0, origin[1]], [0, origin[2]], color='orange', linewidth=2)

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
I1 = 100
I2 = 75
I3 = 80

s_BN_t0 = np.array([0.1, 0.2, -0.1])  # Initial MRP attitude of Nano Spacecraft w.r.t. Inertial Frame
w_BN_B_t0 = np.array([30, 10, -20])/180*np.pi  # Initial angular velocity of Nano Spacecraft w.r.t. Inertial Frame expressed in Body Frame in rad/s    

InertiaTensor = np.array([[I1,0,0],[0,I2,0],[0,0,I3]])  # Inertia Tensor of Nano Spacecraft in kg*m^2
K = 5
Pg = 10 
P = np.array([Pg,Pg,Pg])
L = np.array([0.0, 0.0, 0.0])  # constant disturbance torque in N*m   
tstep = 0.1
tmax = 600+tstep
time = np.arange(0, tmax, tstep)
sigma_ref_S = np.zeros((len(time), 3))
omega_ref_S = np.zeros((len(time), 3))

f = 0.03
s1 = 0.1
s2 = 0.2
s3 = -0.3

sigma_ref = np.array([s1*np.sin(f*time), s2*np.cos(f*time), s3*np.sin(f*time*2)]).T
sigma_dot_ref = np.array([s1*f*np.cos(f*time), -s2*f*np.sin(f*time), s3*f*2*np.cos(f*time*2)]).T

for i in range(0,len(time)):
    omega_ref_a = MRP_InvDifferential(sigma_ref[i], sigma_dot_ref[i])
    if i==0:
        omega_ref=omega_ref_a.reshape(1,3)
    else:
        omega_ref=np.vstack((omega_ref,omega_ref_a.reshape(1,3))) 

sigma,omega,angles,sigmaBR,u,H,T = EOM_MRP_Control_Integrator(InertiaTensor,s_BN_t0, w_BN_B_t0, time, L,
                                                              K, P, 1, 1, 1, 1e100, sigma_ref, omega_ref)

snorm = np.zeros((len(time), 1))
for i in range(1, len(time)):
    snorm[i] = np.linalg.norm(sigma[i])

t_eval = 15
index_t_eval = np.argmin(np.abs(time - t_eval))
sigmaBR_teval = sigmaBR[index_t_eval]
print("---------------------------------------------------")
print(f"sigmaBr @ t = {t_eval}, {sigmaBR_teval}")

t_eval = 40
index_t_eval = np.argmin(np.abs(time - t_eval))
sigmaBR_teval = sigmaBR[index_t_eval]
print("---------------------------------------------------")
print(f"sigmaBr @ t = {t_eval}, {sigmaBR_teval}")

PlotSimulation(True,1)