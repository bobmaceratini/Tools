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

