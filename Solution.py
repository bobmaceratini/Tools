import numpy as np

# --- Utilities ---
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def Bmat_MRP(sigma):
    s2 = np.dot(sigma, sigma)
    return (1 - s2) * np.eye(3) + 2 * skew(sigma) + 2 * np.outer(sigma, sigma)

def mrp_shadow_if_needed(sigma):
    n = np.linalg.norm(sigma)
    if n > 1.0:
        return -sigma / (n**2)
    return sigma

def dcm_from_mrp(sigma):
    s2 = np.dot(sigma, sigma)
    I3 = np.eye(3)
    Sx = skew(sigma)
    num = (8 * np.outer(sigma, sigma) - 4 * (1 - s2) * Sx + (1 - s2)**2 * I3)
    den = (1 + s2)**2
    return num / den

# --- Dynamics ---
def attitude_kinematics(sigma, omega):
    # MRP kinematics: sigma_dot = 1/4 * B(sigma) * omega
    return 0.25 * Bmat_MRP(sigma) @ omega

def rigid_body_dynamics(omega, u, I):
    # Euler's equations: I * omega_dot + omega x (I omega) = u + L, with L=0 here
    return np.linalg.solve(I, (u - skew(omega) @ (I @ omega)))

# --- Control ---
def control_u(sigma_BR, omega_BR, omega_BN, I, K, P, omega_RN=np.zeros(3), omega_RN_dot=np.zeros(3), L=np.zeros(3)):
    # u = -K σ_BR - P ω_BR + I(ω̇_RN - ω_BN × ω_RN) + [ω~_BN] I ω_BN - L
    return (-K * sigma_BR
            - P * omega_BR
            + I @ (omega_RN_dot - np.cross(omega_BN, omega_RN))
            + skew(omega_BN) @ (I @ omega_BN)
            - L)

# --- Simulation ---
def simulate_regulator(T=120.0, h=0.01):
    # Inertias
    I = np.diag([100.0, 75.0, 80.0])  # kg m^2
    # Gains
    K = 5.0          # Nm
    P = 10.0         # Nms (scalar times I_3)
    # Initial states
    sigma_BN = np.array([0.1, 0.2, -0.1], dtype=float)
    omega_BN = np.deg2rad(np.array([30.0, 10.0, -20.0], dtype=float))  # rad/s

    # Reference (regulation): σ_RN = 0, ω_RN = 0, ω̇_RN = 0, L = 0
    sigma_RN = np.zeros(3)
    omega_RN = np.zeros(3)
    omega_RN_dot = np.zeros(3)
    L = np.zeros(3)

    # Data collection
    t = 0.0
    sigma_norm_at_30 = None

    while t <= T + 1e-12:
        # σ_BR = addMRP(-σ_RN, σ_BN) -> with σ_RN=0, σ_BR = σ_BN
        sigma_BR = sigma_BN.copy()
        sigma_BR = mrp_shadow_if_needed(sigma_BR)

        # ω_BR = ω_BN - C_BR * ω_RN; with ω_RN = 0 -> ω_BR = ω_BN
        C_BR = dcm_from_mrp(sigma_BR)
        omega_BR = omega_BN.copy()

        # Control
        u = control_u(sigma_BR, omega_BR, omega_BN, I, K, P, omega_RN, omega_RN_dot, L)

        # RK4 integration
        def f(state):
            s = state[:3]
            w = state[3:]
            # Recompute BR quantities locally (reference still zero)
            s = mrp_shadow_if_needed(s)
            sdot = attitude_kinematics(s, w)
            wdot = rigid_body_dynamics(w, u, I)
            return np.hstack((sdot, wdot))

        x = np.hstack((sigma_BN, omega_BN))
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x_next = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Shadow set if needed
        sigma_BN = mrp_shadow_if_needed(x_next[:3])
        omega_BN = x_next[3:]

        # Capture norm at 30s
        if sigma_norm_at_30 is None and t + h >= 30.0 - 1e-9:
            sigma_norm_at_30 = np.linalg.norm(sigma_BN)

        t += h

    return sigma_norm_at_30

if __name__ == "__main__":
    val_30s = simulate_regulator(T=120.0, h=0.01)
    print(f"MRP norm at 30 s: {val_30s:.6f}")

