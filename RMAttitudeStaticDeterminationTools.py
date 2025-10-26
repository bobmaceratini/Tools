import numpy as np
from RMKinematicTools import *
import sympy as sp


def vector_normalize(v):
    """
    Normalize a vector.

    Parameters:
    v : array-like
        Input vector.

    Returns:
    v_normalized : ndarray
        Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def tilde(v):
    Vt = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return Vt

def Triad2DCM(v1_b, v2_b, v1_r, v2_r):
    """
    Compute the Direction Cosine Matrix (DCM) using the TRIAD method.

    Parameters:
    v1_b : array-like
        First vector in body frame.
    v2_b : array-like
        Second vector in body frame.
    v1_r : array-like
        First vector in reference frame.
    v2_r : array-like
        Second vector in reference frame.

    Returns:
    DCM : ndarray
        Direction Cosine Matrix from reference frame to body frame.
    """
    t1_b = vector_normalize(v1_b)
    t2_b = vector_normalize(np.cross(t1_b, v2_b))
    t3_b = np.cross(t1_b, t2_b)

    t1_r = vector_normalize(v1_r)
    t2_r = vector_normalize(np.cross(t1_r, v2_r))
    t3_r = np.cross(t1_r, t2_r)

    DCM_b = np.column_stack((t1_b, t2_b, t3_b))
    DCM_r = np.column_stack((t1_r, t2_r, t3_r))

    DCM = DCM_b @ DCM_r.T
    return DCM

def Davenport2Quat(VB, VN, weights):
    B=  np.zeros((3,3))
    for i in range(len(weights)):
        B += weights[i] * np.outer (vector_normalize(VB[i]), vector_normalize(VN[i]))
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([[B[1,2] - B[2,1]],
                   [B[2,0] - B[0,2]],
                   [B[0,1] - B[1,0]]])
    K = np.zeros((4,4))
    K[0,0] = sigma
    K[0,1:4] = Z.T
    K[1:4,0] = Z[:,0]
    K[1:4,1:4] = S - sigma * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eig(K)
    max_index = np.argmax(eigenvalues)
    q_opt = eigenvectors[:, max_index]
    q_opt = q_opt / np.linalg.norm(q_opt)
    q0, q1, q2, q3 = q_opt
    return q_opt, eigenvalues[max_index]

def QUEST2Quat(VB, VN, weights):
    B=  np.zeros((3,3))
    for i in range(len(weights)):
        B += weights[i] * np.outer (vector_normalize(VB[i]), vector_normalize(VN[i]))
    sigma = np.trace(B)
    S = B + B.T
    Z = np.array([[B[1,2] - B[2,1]],
        [B[2,0] - B[0,2]],
        [B[0,1] - B[1,0]]])
    K = np.zeros((4,4))
    K[0,0] = sigma
    K[0,1:4] = Z.T
    K[1:4,0] = Z[:,0]
    K[1:4,1:4] = S - sigma * np.eye(3)

    lam = sp.symbols('lambda')   
    Ks = sp.Matrix(K)
    char_eq = Ks.charpoly(lam)
    a4 = char_eq.all_coeffs()[0]
    a3 = char_eq.all_coeffs()[1]
    a2 = char_eq.all_coeffs()[2]
    a1 = char_eq.all_coeffs()[3]
    a0 = char_eq.all_coeffs()[4]
    

    lamdba_opt = sum(weights)
    for _ in range(4):
        f = (a4*lamdba_opt**4 + a3 * lamdba_opt**3 + a2 * lamdba_opt**2 + a1 * lamdba_opt + a0)
        f_prime = (4*a4 * lamdba_opt**3 + 3 * a3 * lamdba_opt**2 + 2 * a2 * lamdba_opt + a1)
        lamdba_opt = lamdba_opt - f / f_prime

    Matrix = (lamdba_opt + sigma) * np.eye(3) - S
    Matrix_f = Matrix.astype(float)
    Matrix_inv = np.linalg.inv(Matrix_f)
    s = Matrix_inv @ Z
    s=s.squeeze()
    q = CRP2Quat(s)
    return q, lamdba_opt

def OLEA2Quat(VB, VN, weights):
    s = vector_normalize(VB[0]) + vector_normalize(VN[0])
    s = tilde(s)
    d = vector_normalize(VB[0]) - vector_normalize(VN[0])
    d = np.vstack(d)
    S = s
    D = d
    W = np.zeros((3*len(weights),3*len(weights)))
    W[0:3,0:3] = weights[0]*np.eye(3,3)
    for i in range(1,len(weights)):
        s = vector_normalize(VB[i]) + vector_normalize(VN[i])
        s = tilde(s)
        d = vector_normalize(VB[i]) - vector_normalize(VN[i])
        d = np.vstack(d)
        S = np.vstack((S,s))
        D = np.vstack((D,d))
        W[i*3:(i+1)*3,i*3:(i+1)*3] = weights[i]*np.eye(3,3)

    M = S.T @ W @ S
    CRP =  np.linalg.inv(M) @ S.T @ W @ D
    q = CRP2Quat(CRP)
    q = q.squeeze()
    return q


    
        