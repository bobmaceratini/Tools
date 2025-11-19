def IntegratorEulerStep(x0,A,B,u,dt):
    """
    Integrate the state x using Euler integration method.

    Parameters:
    x0 : array-like
        Initial state vector.
    A : 2D array-like
        State matrix.
    B : 2D array-like
        Input matrix.
    u : array-like
        Input vector.
    dt : float
        Time step for integration.

    Returns:
    x1 : ndarray
        State vector after time step dt.
    """
    x1 = x0 + dt * (A @ x0 + B @ u)
    return x1

def IntegratorRK4Step_Function(x0, f, u, dt):
    """
    Integrate the state x using the 4th-order Runge-Kutta method.

    Parameters:
    x0 : array-like
        Initial state vector.
    f : function
        Function that computes the derivative of the state.
        It should have the signature f(x, u) and return dx/dt.
    u : array-like
        Input vector.
    dt : float
        Time step for integration.

    Returns:
    x1 : ndarray
        State vector after time step dt.
    """
    k1 = f(x0, u)
    k2 = f(x0 + 0.5 * dt * k1, u)
    k3 = f(x0 + 0.5 * dt * k2, u)
    k4 = f(x0 + dt * k3, u)

    x1 = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x1

def IntegratorRK4Step(x0, A, B, u, dt):
    """
    Integrate the state x using the 4th-order Runge-Kutta method.

    Parameters:
    x0 : array-like
        Initial state vector.
    A : 2D array-like
        State matrix.
    B : 2D array-like
        Input matrix.
    u : array-like
        Input vector.
    dt : float
        Time step for integration.

    Returns:
    x1 : ndarray
        State vector after time step dt.
    """
    k1 = A @ x0 + B @ u
    k2 = A @ (x0 + 0.5 * dt * k1) + B @ u
    k3 = A @ (x0 + 0.5 * dt * k2) + B @ u
    k4 = A @ (x0 + dt * k3) + B @ u

    x1 = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x1   