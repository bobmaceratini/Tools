import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Parametri iniziali
initial_angles_deg = np.array([40, 30, 80])
initial_angles_rad = np.radians(initial_angles_deg)
deg2rad = np.pi / 180
omega_scale = 20 * deg2rad

# Equazioni differenziali per 3-2-1
def euler_321_kinematics(t, angles):
    psi, theta, phi = angles
    omega_b = omega_scale * np.array([
        np.sin(0.1 * t)+1,
        np.sin(0.1 * t),
        np.sin(0.1 * t)
    ])
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    t_theta = np.tan(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    T = np.array([
        [0, s_phi / c_theta, c_phi / c_theta],
        [0, c_phi, -s_phi],
        [1, s_phi * t_theta, c_phi * t_theta]
    ])
    return T @ omega_b

# Integrazione
t_eval = np.arange(0, 60.1, 0.1)
sol = solve_ivp(euler_321_kinematics, (0, 60), initial_angles_rad, t_eval=t_eval, method='RK45')
angles = sol.y.T  # shape: (N, 3)

# Estrai gli angoli dal risultato dell'integrazione
psi_vals = sol.y[0]
theta_vals = sol.y[1]
phi_vals = sol.y[2]

print(f"Norma degli angoli a t=42s: {np.linalg.norm(sol.y[:, np.where(np.isclose(t_eval, 42.0))[0][0]]):.4f} rad")


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

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=False)
plt.show()
