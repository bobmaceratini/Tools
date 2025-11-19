import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from RMAttitudeStaticDeterminationTools import *
from RMKineticsTools import *
from RMIntegrationTools import *
from RMKinematicTools import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from RMKinematicTools import *


#Initial parameters
f = 0.05*0
s1 = 0.2*0
s2 = 0.3*0
s3 = -0.3*0

sigma0 = np.array([0.1, 0.2, -0.1])*0
#sigma0 = np.array([0.0, 0.0, 0.0])
#inital_W_BM_degps = np.array([0.0, 0.0, 0.0])
inital_W_BM_degps = np.array([30.0, 10.0, -20.0])
omega0 = inital_W_BM_degps/180*np.pi
Inertia_vet = np.array([100.0, 75.0, 80.0])
InertiaTensor = np.array([[Inertia_vet[0],0,0],[0,Inertia_vet[1],0,],[0,0,Inertia_vet[2]]])



print(f"initial MRP: {sigma0}")
print(f"Initial Omega [dag/s]: {inital_W_BM_degps}")
print(f"Inertia Tensor Body Frame:\n {InertiaTensor}")

#Simulation
t_eval = np.arange(0, 40.05, 0.05)

sigma_ref = np.array([s1*np.sin(f*t_eval), s2*np.cos(f*t_eval), s3*np.sin(f*t_eval)]).T

sigma_dot_ref = np.array([s1*f*np.cos(f*t_eval), -s2*f*np.sin(f*t_eval), s3*f*np.cos(f*t_eval)]).T

for i in range(0,len(t_eval)):
    omega_ref_a = MRP_InvDifferential(sigma_ref[i], sigma_dot_ref[i])
    if i==0:
        omega_ref=omega_ref_a.reshape(1,3)
    else:
        omega_ref=np.vstack((omega_ref,omega_ref_a.reshape(1,3))) 

sigma,omega,omegamis,angles,sigmaBR,u,z = EOM_Rotation_RMP_Integrator_KI(InertiaTensor,sigma0, omega0, t_eval)

t= 2

print(f"Norma of sigma diff whtn t={t}s:  {np.linalg.norm(sigmaBR[np.where(np.isclose(t_eval,t))[0][0]]):.4f}")


psi_vals = angles[:,0]
theta_vals = angles[:,1]
phi_vals = angles[:,2]


# Crea il grafico
plt.figure(figsize=(10, 6))
plt.plot(t_eval, psi_vals, label='Yaw (ψ)', color='blue')
plt.plot(t_eval, theta_vals, label='Pitch (θ)', color='green')
plt.plot(t_eval, phi_vals, label='Roll (ϕ)', color='orange')

plt.xlabel('Tempo [s]')
plt.ylabel('Angoli [rad]')
plt.title('Evoluzione degli angoli di Eulero (3-2-1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Crea il grafico
plt.figure(figsize=(10, 6))
plt.plot(t_eval, sigma[:,0], label='sigma(1)', color='blue')
plt.plot(t_eval, sigma_ref[:,0], '--', label='sigma\_ref(1)', color='blue')
plt.plot(t_eval, sigma[:,1], label='sigma(2)', color='green')
plt.plot(t_eval, sigma_ref[:,1], '--', label='sigma\_ref(2)', color='green')
plt.plot(t_eval, sigma[:,2], label='sigma(3)', color='orange')
plt.plot(t_eval, sigma_ref[:,2], '--', label='sigma\_ref(3)', color='orange')

plt.xlabel('Tempo [s]')
plt.ylabel('Angoli [rad]')
plt.title('Evoluzione componenti MRP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_eval, omegamis[:,0]/np.pi*180, label='omega(1)', color='blue')
plt.plot(t_eval, omegamis[:,1]/np.pi*180, label='omega(2)', color='green')
plt.plot(t_eval, omegamis[:,2]/np.pi*180, label='omega(3)', color='orange')

plt.xlabel('Time [s]')
plt.ylabel('Omega [deg/s]')
plt.title('Angular Speed Components, measurements with noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sdiff = np.zeros((len(t_eval), sigma0.size))
snorm = np.zeros((len(t_eval), 1))

for i in range(1, len(t_eval)):
    sdiff[i]=-MRPSum(-sigma[i],sigma_ref[i])
    snorm[i] = np.linalg.norm(sigmaBR[i])
    if np.linalg.norm(sdiff[i])>1:
        sdiff[i]=MRP2Shadow(sdiff[i])


plt.figure(figsize=(10, 6))
plt.plot(t_eval, sigmaBR[:,0], label='sigmaBR(1)', color='blue')
plt.plot(t_eval, sdiff[:,0], '--', label='sigma\_diff(1)', color='blue')
plt.plot(t_eval, sigmaBR[:,1], label='sigmaBR(2)', color='green')
plt.plot(t_eval, sdiff[:,1], '--', label='sigma\_diff(2)', color='green')
plt.plot(t_eval, sigmaBR[:,2], label='sigmaBR(3)', color='orange')
plt.plot(t_eval, sdiff[:,2], '--', label='sigma\_diff(3)', color='orange')
plt.plot(t_eval, snorm, label='sigma', color='red')
plt.plot(t_eval, snorm, label='u', color='red')


plt.xlabel('Tiome [s]')
plt.ylabel('m')
plt.title('MRP components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_eval, u[:,0], label='u(1)', color='blue')
plt.plot(t_eval, u[:,1], label='u(2)', color='green')
plt.plot(t_eval, u[:,2], label='u(3)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Control Torque [Nm]')
plt.title('Control Torque Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
lines = [ax.plot([], [], [], 'b')[0] for _ in edges]


def update(frame):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    psi, theta, phi = angles[frame]
    rot = R.from_euler('zyx', [psi, theta, phi])
    rotated_vertices = rot.apply(cube_vertices)

    # Disegna il cubo
    for start, end in edges:
        xs, ys, zs = zip(rotated_vertices[start], rotated_vertices[end])
        ax.plot(xs, ys, zs, 'b')

    # Terna solidale (assi del corpo)
    origin = np.array([0, 0, 0])
    body_axes = np.eye(3)  # X, Y, Z unitari
    rotated_axes = rot.apply(body_axes)

    colors = ['r', 'b', 'g']  # tutte rosse
    labels = ['X_body', 'Y_body', 'Z_body']

    for vec, color, label in zip(rotated_axes, colors, labels):
        ax.quiver(*origin, *vec, color=color, length=1.5, normalize=True)
        ax.text(*(vec * 1.6), label, color=color)

    ax.set_title(f"t = {t_eval[frame]:.1f}s")

    return lines

skip = 2  # Salta frame per velocizzare l'animazione
ani = FuncAnimation(fig, update, frames=range(0, len(t_eval), skip), interval=10, blit=False)
plt.show()

