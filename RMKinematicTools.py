import numpy as np

# Rotation Matrices and Quaternion Transformations
# List of functions for converting between different attitude representations
# including Direction Cosine Matrices (DCM), Quaternions, Euler Angles, Classical Rodrigues Parameters (CRP), and Modified Rodrigues Parameters (MRP).
#   Each function is documented with its purpose and parameters.
#   Author: OpenAI's ChatGPT and toberto Maceratini
#   Date: 2024-06
#   Version: 1.0
#   License: MIT License
#   Note: This code requires NumPy library.
#   Usage: Import this module and call the desired functions for attitude representation conversions.
#   Example: DCM = Quat2DCM(q) converts quaternion q to direction cosine matrix DCM.
#   Disclaimer: This code is provided "as is" without warranty of any kind.
#   For educational and illustrative purposes only.
#   Do not use in safety-critical applications.
#   Always verify results independently.
#   End of header.
#       
#   Functions:
#   - Quat2DCM
#   - DCM2Quat
#   - Eu2DCM_ZYX
#   - DCM2Eu_ZYX
#   - Eu2DCM_ZXZ
#   - DCM2Eu_ZXZ
#   - QuatSum
#   - QuatSub   
#   - QuatFlip
#   - Eu2Quat_ZYX
#   - Quat2Eu_ZYX
#   - CRP2DCM
#   - DCM2CRP
#   - Quat2CRP  
#   - CRP2Quat
#   - MRP2Shadow
#   - Quat2MRP
#   - MRP2Quat


def Eu2DCM_ZYX(ea):
    teta1, teta2, teta3 = ea
    c1 = np.cos(teta1)
    c2 = np.cos(teta2)
    c3 = np.cos(teta3)
    s1 = np.sin(teta1)
    s2 = np.sin(teta2)
    s3 = np.sin(teta3)
    DCM = np.array([
        [c2*c1,          c2*s1,             -s2],   
        [s3*s2*c1-c3*s1, s3*s2*s1+c3*c1,  s3*c2],
        [c3*s2*c1+s3*s1, c3*s2*s1-s3*c1,        c3*c2]
    ])
    return DCM

def DCM2Eu_ZYX(DCM):
    teta2 = -np.arcsin(DCM[0,2])
    teta1 = np.arctan2(DCM[0,1], DCM[0,0])
    teta3 = np.arctan2(DCM[1,2], DCM[2,2])
    return np.array([teta1, teta2, teta3])  

def Eu2DCM_ZXZ(ea):
    teta1, teta2, teta3 = ea
    c1 = np.cos(teta1)
    c2 = np.cos(teta2)
    c3 = np.cos(teta3)
    s1 = np.sin(teta1)
    s2 = np.sin(teta2)
    s3 = np.sin(teta3)
    DCM = np.array([
        [c3*c1-s3*c2*s1, c3*s1+s3*c2*c1, s3*s2],
        [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
        [s2*s1, -s2*c1, c2] 
   
    ])
    return DCM

def DCM2Eu_ZXZ(DCM):
    teta2 = np.arccos(DCM[2,2])
    teta1 = np.atan2(DCM[2,0], -DCM[2,1])
    teta3 = np.atan2(DCM[0,2], DCM[1,2])
    return np.array([teta1, teta2, teta3])  

def DCM2PRV(DCM):
    angle = np.arccos((np.trace(DCM) - 1) / 2)
    if abs(angle) < 1e-6:
        return np.zeros(3)
    elif abs(angle - np.pi) < 1e-6:
        R = DCM
        r1 = np.sqrt((R[0,0] + 1) / 2)
        r2 = np.sqrt((R[1,1] + 1) / 2)
        r3 = np.sqrt((R[2,2] + 1) / 2)
        if R[0,1] < 0:
            r2 = -r2
        if R[0,2] < 0:
            r3 = -r3
        axis = np.array([r1, r2, r3])
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([
            DCM[2,1] - DCM[1,2],
            DCM[0,2] - DCM[2,0],
            DCM[1,0] - DCM[0,1]
        ]) / (2 * np.sin(angle))
    prv = axis * angle
    return prv
    

def Quat2DCM(q):
    """
    Convert a quaternion to a direction cosine matrix (DCM).

    Parameters:
    q : array-like
        A quaternion represented as a list or array of four elements [q0, q1, q2, q3],
        where q0 is the scalar part and [q1, q2, q3] is the vector part.

    Returns:
    DCM : ndarray
        A 3x3 direction cosine matrix corresponding to the input quaternion.
    """
    import numpy as np

    q0, q1, q2, q3 = q

    DCM = np.array([
        [(q0**2+q1**2-q2**2-q3**2),     2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)],
        [    2*(q1*q2 - q0*q3), (q0**2-q1**2+q2**2-q3**2),     2*(q2*q3 + q0*q1)],
        [    2*(q3*q1 + q0*q2),     2*(q3*q2 - q0*q1), (q0**2-q1**2-q2**2+q3**2)]
    ])

    return DCM

def DCM2Angle(DCM):
    """
    Compute the angle of rotation represented by a direction cosine matrix (DCM).

    Parameters:
    DCM : ndarray
        A 3x3 direction cosine matrix.

    Returns:
    angle : float
        The angle of rotation in radians.
    """
    s = DCM2MRP(DCM)
    if np.linalg.norm(s) > 1:
        s = MRP2Shadow(s)
    angle = 4 * np.arctan(np.linalg.norm(s))
    return angle

def DCM2Quat(DCM):
    """
    Convert a direction cosine matrix (DCM) to a quaternion.

    Parameters:
    DCM : ndarray
        A 3x3 direction cosine matrix.

    Returns:
    q : ndarray
        A quaternion represented as an array of four elements [q0, q1, q2, q3],
        where q0 is the scalar part and [q1, q2, q3] is the vector part.
    """
    import numpy as np

    m = DCM
    tr = np.trace(m)

    b0_sq = 0.25*(tr + 1)
    b1_sq = 0.25*(1 + 2*m[0,0   ] - tr)
    b2_sq = 0.25*(1 + 2*m[1,1   ] - tr)
    b3_sq = 0.25*(1 + 2*m[2,2   ] - tr)
    b = np.array([b0_sq, b1_sq, b2_sq, b3_sq])  
    max_index = np.argmax(b)
    if max_index == 0:
        q0 = np.sqrt(b0_sq)
        q1 = (-m[2,1] + m[1,2])/(4*q0)
        q2 = (-m[0,2] + m[2,0])/(4*q0)
        q3 = (-m[1,0] +  m[0,1])/(4*q0)
    elif max_index == 1:
        q1 = np.sqrt(b1_sq)
        q0 = (-m[2,1] + m[1,2])/(4*q1)
        q2 = (+m[0,1] + m[1,0])/(4*q1)
        q3 = (m[0,2] + m[2,0])/(4*q1)
    elif max_index == 2:
        q2 = np.sqrt(b2_sq)
        q0 = (-m[0,2] + m[2,0])/(4*q2)
        q1 = (m[0,1] + m[1,0])/(4*q2)
        q3 = (m[1,2] + m[2,1])/(4*q2)
    else:  # max_index == 3
        q3 = np.sqrt(b3_sq)
        q0 = (-m[1,0] + m[0,1])/(4*q3)
        q1 = (m[0,2] + m[2,0])/(4*q3)
        q2 = (m[1,2] + m[2,1])/(4*q3)

    q = np.array([q0, q1, q2, q3])
    return q

def ZYX2DCM(eu):
    teta1, teta2, teta3 = eu    
    c1 = np.cos(teta1)
    c2 = np.cos(teta2)    
    c3 = np.cos(teta3)
    s1 = np.sin(teta1)
    s2 = np.sin(teta2)
    s3 = np.sin(teta3)
    DCM = np.array([
        [c2*c1,          c2*s1,             -s2],
        [s3*s2*c1-c3*s1, s3*s2*s1+c3*c1,  s3*c2],
        [c3*s2*c1+s3*s1, c3*s2*s1-s3*c1,        c3*c2]
    ])
    return DCM

def ZYX2Quat2(eu):
    teta1, teta2, teta3 = eu    
    c1 = np.cos(teta1/2)
    c2 = np.cos(teta2/2)    
    c3 = np.cos(teta3/2)
    s1 = np.sin(teta1/2)
    s2 = np.sin(teta2/2)
    s3 = np.sin(teta3/2)
    q0 = c1*c2*c3 + s1*s2*s3
    q1 = s1*c2*c3 - c1*s2*s3        
    q2 = c1*s2*c3 + s1*c2*s3
    q3 = c1*c2*s3 - s1*s2*c3    
    return np.array([q0, q1, q2, q3])

def ZYZ2Quat(eu):
    DCM = Eu2DCM_ZYX(eu)
    return DCM2Quat(DCM)

def QuatSum(q1, q2):
    """
    Compute the composition of two quaternions.

    Parameters:
    q1 : array-like
        First quaternion [q0, q1, q2, q3].
    q2 : array-like
        Second quaternion [q0, q1, q2, q3].

    Returns:
    q_comp : ndarray
        The resulting quaternion from the composition of q1 and q2.
    """

    B = np.array([[ q2[0], -q2[1], -q2[2], -q2[3]],
                  [ q2[1],  q2[0],  q2[3], -q2[2]],
                  [ q2[2], -q2[3],  q2[0],  q2[1]],
                  [ q2[3],  q2[2], -q2[1],  q2[0]]] )
    
    q_comp = B @ q1
    return q_comp

def QuatSub(q1, q2):
    """
    Compute the inverse composition of two quaternions.

    Parameters:
    q1 : array-like
        First quaternion [q0, q1, q2, q3].
    q2 : array-like
        Second quaternion [q0, q1, q2, q3].

    Returns:
    q_inv_comp : ndarray
        The resulting quaternion from the inverse composition of q1 and q2.
    """

    B = np.array([[ q2[0],  q2[1],  q2[2],  q2[3]],
                  [-q2[1],  q2[0],  q2[3], -q2[2]],
                  [-q2[2], -q2[3],  q2[0],  q2[1]],
                  [-q2[3],  q2[2], -q2[1],  q2[0]]] )
    
    q_inv_comp = B @ q1
    return q_inv_comp

def QuatFlip(q):

    return (-q)
    return (-q)

def Eu2Quat_ZYX(eu):
    DCM = Eu2DCM_ZYX(eu)
    return DCM2Quat(DCM)

def Quat2Eu_ZYX(q):
    DCM = Quat2DCM(q)
    return DCM2Eu_ZYX(DCM)

def CRP2DCM(sigma):
    sigma1, sigma2, sigma3 = sigma
    s2 = sigma1**2 + sigma2**2 + sigma3**2
    snorm = np.sqrt(s2)
    DCM = 1/(1+s2)*np.array([[1+sigma1**2 - sigma2**2 - sigma3**2, 2*(sigma1*sigma2 + sigma3), 2*(sigma1*sigma3 - sigma2)],
                             [2*(sigma2*sigma1 - sigma3), 1 - sigma1**2 + sigma2**2 - sigma3**2, 2*(sigma2*sigma3 + sigma1)],
                             [2*(sigma3*sigma1 + sigma2), 2*(sigma3*sigma2 - sigma1), 1 - sigma1**2 - sigma2**2 + sigma3**2]])       
    return DCM

def DCM2CRP(DCM):
    m = DCM
    tr = np.trace(m)
    s2 = (1 - tr)/(1 + tr)
    s = np.sqrt(s2)
    sigma1 = (m[1,2] - m[2,1])/(tr + 1)
    sigma2 = (m[2,0] - m[0,2])/(tr + 1)
    sigma3 = (m[0,1] - m[1,0])/(tr + 1)
    return np.array([sigma1, sigma2, sigma3])

def Quat2CRP(q):
    s1 = q[1]/(1 + q[0])
    s2 = q[2]/(1 + q[0])    
    s3 = q[3]/(1 + q[0])
    return np.array([s1, s2, s3])

def CRP2Quat(sigma):
    snorm2 = np.dot(sigma.T, sigma)
    q0 = 1/np.sqrt(1 + snorm2)
    q1 = sigma[0]*q0
    q2 = sigma[1]*q0
    q3 = sigma[2]*q0
    return np.array([q0, q1, q2, q3])

def MRP2Shadow(sigma):
    s2 = np.dot(sigma, sigma)
    shadow_sigma = -sigma/(s2)
    return shadow_sigma

def Quat2MRP(q):
    s1 = q[1]/(1 + q[0])
    s2 = q[2]/(1 + q[0])    
    s3 = q[3]/(1 + q[0])
    return np.array([s1, s2, s3])

def MRP2Quat(sigma):
    snorm2 = np.dot(sigma, sigma)
    q0 = (1 - snorm2)/(1 + snorm2)
    q1 = (2*sigma[0])/(1 + snorm2)
    q2 = (2*sigma[1])/(1 + snorm2)
    q3 = (2*sigma[2])/(1 + snorm2)
    return np.array([q0, q1, q2, q3])

def MRP2DCM(sigma):
    sigma1, sigma2, sigma3 = sigma
    s2 = sigma1**2 + sigma2**2 + sigma3**2
    Eye = np.eye(3)
    sigma_vec = np.array([[0, -sigma3, sigma2],
                            [sigma3, 0, -sigma1],
                            [-sigma2, sigma1, 0]])  
    DCM = Eye + (8*sigma_vec @ sigma_vec - 4*(1 - s2)*sigma_vec)/(1 + s2)**2    
    return DCM

def DCM2MRP(DCM):
    q = DCM2Quat(DCM)
    s = Quat2MRP(q)
    return s

def MRP2EU_ZYX(sigma):
    q = MRP2Quat(sigma)
    angles = Quat2Eu_ZYX(q)
    return angles


def MRPSum(s1,s2):
    s1_sq = np.dot(s1,s1)
    s2_sq = np.dot(s2,s2)
    denominator = 1 + s1_sq*s2_sq - 2*np.dot(s1,s2)
    if abs(denominator) < 0.0001:
        s1=MRP2Shadow(s1)
        s1_sq = np.dot(s1,s1)
        s2_sq = np.dot(s2,s2)
        denominator = 1 + s1_sq*s2_sq - 2*np.dot(s1,s2)
        
    numerator = (1 - s1_sq)*s2 + (1 - s2_sq)*s1 - 2*np.cross(s2,s1)
    s_comp = numerator/denominator
    return s_comp

def MRPSub(s1,s):
    s1_sq = np.dot(s1,s1)
    s_sq = np.dot(s,s)
    denominator = 1 + s1_sq*s_sq + 2*np.dot(s1,s)
    if abs(denominator) < 0.0001:
        s1 = MRP2Shadow(s1)
        s1_sq = np.dot(s1,s1)
        s_sq = np.dot(s,s)
        denominator = 1 + s1_sq*s_sq + 2*np.dot(s1,s)
        
    numerator = (1 - s1_sq)*s - (1 - s_sq)*s1 + 2*np.cross(s,s1)
    s_comp = numerator/denominator
    return s_comp

def ZYX_Differential(angles, omega_b):
    psi, theta, phi = angles
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

def Quat_Differential(q, omega):
    Omega = np.array([
        [0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0, omega[0]],
        [omega[2], omega[1], -omega[0], 0]
    ])
    dqdt = 0.5 * Omega @ q
    return dqdt

def CRP_Differential(sig, omega):
    sig1, sig2, sig3 = sig
    B = 0.5*np.array([[1+sig1**2,sig1*sig2 - sig3,sig1*sig3 + sig2],
                      [sig2*sig1 + sig3,1 + sig2**2,sig2*sig3 - sig1],
                      [sig3*sig1 - sig2,sig3*sig2 + sig1,1 + sig3**2]])
    dsigdt =  B @ omega
    return dsigdt

def MRP_Differential(sig, omega):
    sig1, sig2, sig3 = sig
    s2 = sig1**2 + sig2**2 + sig3**2
    skew = np.array([[0, -sig3, sig2],
                     [sig3, 0, -sig1],
                        [-sig2, sig1, 0]])
    
    B = 0.25*((1 - s2)*np.eye(3) + 2*skew + 2*np.outer(sig, sig))
    dsigdt = B @ omega
    return dsigdt   

def MRP_InvDifferential(sigmam,sig_dot):
    s2 = np.dot(sigmam,sigmam)
    skew = np.array([[0, -sigmam[2], sigmam[1]],
                     [sigmam[2], 0, -sigmam[0]],
                        [-sigmam[1], sigmam[0], 0]])
    
    B_inv = 4/((1 + s2)**2)*((1 - s2)*np.eye(3) - 2*skew + 2*np.outer(sigmam, sigmam))
    omega = B_inv @ sig_dot
    return omega

def ZYZ_Differential(angles, angles_dot):
    teta1, teta2, teta3 = angles
    s2 = np.sen(teta2)
    c2 = np.cos(teta2)
    s3 = np.sin(teta3)
    c3 = np.cos(teta3)
    T = np.array([
        [s3 * s2, c3, 0],
        [c3*s2 , -s3, 0],
        [c2, 0, 1]
    ])
    return T @ angles_dot  

    

def Kinetic_Integration(differential_func, omega_func,t_span, x0, t_eval):
    N=len(t_eval)
    x = np.zeros((N, x0.size))
    x[0] = x0
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        k1 = differential_func(t_eval[t_index - 1], x[t_index - 1], omega_func(t_eval[t_index - 1]))
        x[t_index] = x[t_index - 1] + dt * k1
    return x

def Kinetic_IntegrationSat(differential_func, omega_func,t_span, x0, t_eval):
    N=len(t_eval)
    x = np.zeros((N, x0.size))
    x[0] = x0
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        k1 = differential_func(t_eval[t_index - 1], x[t_index - 1], omega_func(t_eval[t_index - 1]))
        x[t_index] = x[t_index - 1] + dt * k1
        if np.linalg.norm(x[t_index]) > 1:
            x[t_index] = MRP2Shadow(x[t_index])
            
    return x