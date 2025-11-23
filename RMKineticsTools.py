import numpy as np
from RMKinematicTools import *
from RMAttitudeStaticDeterminationTools import *

class MassPoint:
    def __init__(self, position_comp, velocity_comp, mass):
        self.position_v = position_comp
        self.velocity_v = velocity_comp
        self.mass = mass

    def __repr__(self):
        return f"Position: {self.position_v}, Velocity:  {self.velocity_v}, Mass: {self.mass}\n"
    
def CalcCenterOfMass(MassPointArray):
    CDM =  MassPoint(np.array([0,0,0]),np.array([0,0,0]),0)
    for point in MassPointArray:
        CDM.mass = CDM.mass + point.mass
        CDM.position_v = CDM.position_v + point.position_v*point.mass
        CDM.velocity_v = CDM.velocity_v + point.velocity_v*point.mass
    
    CDM.position_v = CDM.position_v/CDM.mass
    CDM.velocity_v = CDM.velocity_v/CDM.mass

    return CDM
    
def CalcKineticEnergy(MassPointArray):
    T_tot = 0
    for point in MassPointArray:
        p_n = point.velocity_v.T @ point.velocity_v
        T_tot = T_tot + 0.5*point.mass*p_n
    
    return T_tot

def CalcAngularMomentum(MassPointArray,P):
    H = np.array([0,0,0])
    for point in MassPointArray:
        rel_pos = point.position_v-P.position_v
        rel_velocity = point.velocity_v - P.velocity_v
        H = H + np.cross(rel_pos,rel_velocity)*point.mass

    return H

def EOM_Rotation(InertiaTensor,invInertiaTensor,omega,u,L):
    w_tilde = tilde(omega)
    deltaOmega = invInertiaTensor @ (-w_tilde @ InertiaTensor @ omega + u + L)
    return deltaOmega


def EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,sigma,omega,u,L):
    deltasigma = MRP_Differential(sigma,omega)
    deltaomega = EOM_Rotation(InertiaTensor,invInertiaTensor,omega,u,L)
    return deltasigma,deltaomega
    
def EOM_Rotation_RMP_Integrator(InertiaTensor,sigma0, omega0, t_eval,sigma_ref=None,omega_ref=None):
    N=len(t_eval)
    invInertiaTensor = np.linalg.inv(InertiaTensor)
    sigma = np.zeros((N, sigma0.size))
    omega = np.zeros((N, sigma0.size))
    angles = np.zeros((N, sigma0.size))
    sigmaBR = np.zeros((N, sigma0.size))
    u = np.zeros((N, sigma0.size))
    sigma[0] = sigma0
    omega[0] = omega0
    angles[0]=MRP2EU_ZYX(sigma0)
    K = 5
    P = 10
    if (sigma_ref is not None):
        sigmaBR[0] = MRPSum(sigma0,-sigma_ref[0])
        sigmaBR[0] = sigma[0]-sigma_ref[0]
    else:
        sigmaBR[0] = sigma0
        
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        L = np.array([0.5,-0.3,0.2])
        if sigma_ref is not None:
            if np.linalg.norm(sigma[t_index-1]) > 1:
                sigma[t_index-1] = MRP2Shadow(sigma[t_index-1])
            if np.linalg.norm(sigma_ref[t_index-1]) > 1:
                sigma_ref[t_index-1] = MRP2Shadow(sigma_ref[t_index-1])
            sigmaBR[t_index - 1] = -MRPSum(-sigma[t_index - 1],sigma_ref[t_index - 1])
            if np.linalg.norm(sigmaBR[t_index-1]) > 1:
                sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])
            #sigmaBR[t_index-1] = sigma[t_index - 1]-sigma_ref[t_index - 1]
        else: 
            sigmaBR[t_index - 1] = sigma[t_index - 1]

        if np.linalg.norm(sigmaBR[t_index-1]) > 1:
            sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])

        BR = MRP2DCM(sigmaBR[t_index - 1])
        omega_ref_BR = BR @ omega_ref[t_index - 1] if omega_ref is not None else np.array([0,0,0])
        omega_ref_dot_BR = BR @ ((omega_ref[t_index]-omega_ref[t_index-1])/dt) if omega_ref is not None else np.array([0,0,0])

        term_1= - K * (sigmaBR[t_index - 1]) 

        if (omega_ref is not None):
            term_2 = - P * (omega[t_index - 1] - omega_ref_BR)
            term_3 = InertiaTensor@(omega_ref_dot_BR-np.cross(omega[t_index-1],omega_ref_BR))
        else:   
            term_2 = - P * omega[t_index - 1]
            term_3 = 0

        term_4 = tilde(omega[t_index - 1]) @ InertiaTensor @ omega[t_index - 1]

        term_5 = -L

        u[t_index-1]= term_1 + term_2 + term_3 + term_4 + term_5

        k1s,k1o  = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1],omega[t_index-1],u[t_index-1],L)

        k2s,k2o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1]+0.5*dt*k1s,
                                                              omega[t_index-1]+0.5*dt*k1o,u[t_index-1],L)
        k3s,k3o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                                sigma[t_index-1]+0.5*dt*k2s,
                                                                omega[t_index-1]+0.5*dt*k2o,u[t_index-1],L)    
        k4s,k4o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,     
                                                                sigma[t_index-1]+dt*k3s,
                                                                omega[t_index-1]+dt*k3o,u[t_index-1],L)
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        #deltasigma =  k1s
        #deltaomega = k1o     
        sigma[t_index] = sigma[t_index - 1] + dt * deltasigma
        omega[t_index] = omega[t_index-1] + dt * deltaomega
        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])
        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])

    return sigma,omega,angles,sigmaBR,u

def EOM_Rotation_RMP_Integrator_KI(InertiaTensor,sigma0, omega0, t_eval,sigma_ref=None,omega_ref=None):
    umax = 2
    N=len(t_eval)
    D = 0.5/180*np.pi

    invInertiaTensor = np.linalg.inv(InertiaTensor)
    sigma = np.zeros((N, sigma0.size))
    omega = np.zeros((N, sigma0.size))
    omega_mis = np.zeros((N, sigma0.size))
    angles = np.zeros((N, sigma0.size))
    sigmaBR = np.zeros((N, sigma0.size))
    zi = np.zeros((N, sigma0.size))
    z = np.zeros((N, sigma0.size))
    u = np.zeros((N, sigma0.size))
    sigma[0] = sigma0
    omega[0] = omega0
    angles[0]=MRP2EU_ZYX(sigma0)


    K = 5*0
    P1,P2,P3 = 10e9,10e9,10e9
    P = np.array([[P1,0,0],[0,P2,0],[0,0,P3]])
    KI = 0.005*0
    if (sigma_ref is not None):
        sigmaBR[0] = MRPSum(sigma0,-sigma_ref[0])
        sigmaBR[0] = sigma[0]-sigma_ref[0]
    else:
        sigmaBR[0] = sigma0
        
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        L = np.array([0.5,-0.3,0.2])*0

        if sigma_ref is not None:
            if np.linalg.norm(sigma[t_index-1]) > 1:
                sigma[t_index-1] = MRP2Shadow(sigma[t_index-1])
            if np.linalg.norm(sigma_ref[t_index-1]) > 1:
                sigma_ref[t_index-1] = MRP2Shadow(sigma_ref[t_index-1])
            sigmaBR[t_index - 1] = -MRPSum(-sigma[t_index - 1],sigma_ref[t_index - 1])
            if np.linalg.norm(sigmaBR[t_index-1]) > 1:
                sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])
            #sigmaBR[t_index-1] = sigma[t_index - 1]-sigma_ref[t_index - 1]
        else: 
            sigmaBR[t_index - 1] = sigma[t_index - 1]

        if np.linalg.norm(sigmaBR[t_index-1]) > 1:
            sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])

        BR = MRP2DCM(sigmaBR[t_index - 1])
        BR0 = MRP2DCM(sigmaBR[0])

        omega_ref_BR_0 = BR0 @ omega_ref[0] if omega_ref is not None else np.array([0,0,0])
        omega_0 = omega[0]

        omega_ref_BR = BR @ omega_ref[t_index - 1] if omega_ref is not None else np.array([0,0,0])
        omega_ref_dot_BR = BR @ ((omega_ref[t_index]-omega_ref[t_index-1])/dt) if omega_ref is not None else np.array([0,0,0])

        term_1= - K * (sigmaBR[t_index - 1]) 
        x = np.random.normal(0, 0.1, size=(3))
        omega_mis[t_index - 1] = omega[t_index - 1] + x/180*np.pi
        if (omega_ref is not None):
            DeltaOmega =  omega_mis[t_index - 1]  - omega_ref_BR*0
            term_2 = - P @ DeltaOmega
            term_3 = InertiaTensor@(omega_ref_dot_BR-np.cross(omega[t_index-1],omega_ref_BR))
        else:   
            DeltaOmega =  omega_mis[t_index - 1] 
            term_2 = - P @ DeltaOmega
            term_3 = 0

        term_4 = tilde(omega_mis[t_index - 1]) @ InertiaTensor @ omega[t_index - 1]

        term_5 = -L

        term_6 = - P@ (KI * z[t_index - 1])


        #u[t_index-1]= term_1 + term_2 + term_3*0 + term_4*0 + term_5*0 + term_6*0
        G = 0
        if (DeltaOmega[0]>D):
            u[t_index-1][0]=-umax
        else:
            if (DeltaOmega[0]<-D):
                u[t_index-1][0]=umax
            else:
                u[t_index-1][0]=u[t_index-1][0]*G
        if (DeltaOmega[1]>D):
            u[t_index-1][1]=-umax
        else:
            if (DeltaOmega[1]<-D):
                u[t_index-1][1]=umax
            else:
                u[t_index-1][1]=u[t_index-1][1]*G
        if (DeltaOmega[2]>D):
            u[t_index-1][2]=-umax
        else:
            if (DeltaOmega[2]<-D):
                u[t_index-1][2]=umax
            else:
                u[t_index-1][2]=u[t_index-1][2]*G

        if (u[t_index-1][0]>umax):
            u[t_index-1][0]=umax
        if (u[t_index-1][1]>umax):
            u[t_index-1][1]=umax
        if (u[t_index-1][2]>umax):
            u[t_index-1][2]=umax

        if (u[t_index-1][0]<-umax):
             u[t_index-1][0]=-umax
        if (u[t_index-1][1]<-umax):
             u[t_index-1][1]=-umax
        if (u[t_index-1][2]<-umax):
             u[t_index-1][2]=-umax



        k1s,k1o  = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1],omega[t_index-1],u[t_index-1],L)

        k2s,k2o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1]+0.5*dt*k1s,
                                                              omega[t_index-1]+0.5*dt*k1o,u[t_index-1],L)
        k3s,k3o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                                sigma[t_index-1]+0.5*dt*k2s,
                                                                omega[t_index-1]+0.5*dt*k2o,u[t_index-1],L)    
        k4s,k4o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,     
                                                                sigma[t_index-1]+dt*k3s,
                                                                omega[t_index-1]+dt*k3o,u[t_index-1],L)
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        #deltasigma =  k1s
        #deltaomega = k1o     
        sigma[t_index] = sigma[t_index - 1] + dt * deltasigma
        omega[t_index] = omega[t_index-1] + dt * deltaomega

        zi[t_index] = zi[t_index-1] + K*sigmaBR[t_index - 1] * dt
        z[t_index] = zi[t_index] + InertiaTensor @( (omega[t_index ] - omega_ref_BR) - (omega_0-omega_ref_BR_0) )

        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])
        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])

    return sigma,omega,omega_mis,angles,sigmaBR,u,z

def Control_Linear_CLD_MPR(InertiaTensor,sigma0, omega0, K, P, umax, t_eval,sigma_ref=None,omega_ref=None):
    N=len(t_eval)
    invInertiaTensor = np.linalg.inv(InertiaTensor)
    sigma = np.zeros((N, sigma0.size))
    omega = np.zeros((N, sigma0.size))
    angles = np.zeros((N, sigma0.size))
    sigmaBR = np.zeros((N, sigma0.size))
    u = np.zeros((N, sigma0.size))
    sigma[0] = sigma0
    omega[0] = omega0
    angles[0]=MRP2EU_ZYX(sigma0)

    if (sigma_ref is not None):
        sigmaBR[0] = MRPSum(sigma0,-sigma_ref[0])
        sigmaBR[0] = sigma[0]-sigma_ref[0]
    else:
        sigmaBR[0] = sigma0
        
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        L = np.array([0.5,-0.3,0.2])*0
        if sigma_ref is not None:
            if np.linalg.norm(sigma[t_index-1]) > 1:
                sigma[t_index-1] = MRP2Shadow(sigma[t_index-1])
            if np.linalg.norm(sigma_ref[t_index-1]) > 1:
                sigma_ref[t_index-1] = MRP2Shadow(sigma_ref[t_index-1])
            sigmaBR[t_index - 1] = -MRPSum(-sigma[t_index - 1],sigma_ref[t_index - 1])
            if np.linalg.norm(sigmaBR[t_index-1]) > 1:
                sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])
                #sigmaBR[t_index-1] = sigma[t_index - 1]-sigma_ref[t_index - 1]
        else: 
            sigmaBR[t_index - 1] = sigma[t_index - 1]

        if np.linalg.norm(sigmaBR[t_index-1]) > 1:
            sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])

        BR = MRP2DCM(sigmaBR[t_index - 1])
        BR0 = MRP2DCM(sigmaBR[0])

        omega_ref_BR_0 = BR0 @ omega_ref[0] if omega_ref is not None else np.array([0,0,0])
        omega_0 = omega[0]

        omega_ref_BR = BR @ omega_ref[t_index - 1] if omega_ref is not None else np.array([0,0,0])
        omega_ref_dot_BR = BR @ ((omega_ref[t_index]-omega_ref[t_index-1])/dt) if omega_ref is not None else np.array([0,0,0])

        WW = np.dot(omega[t_index - 1].T,omega[t_index - 1])
        W2 = np.linalg.norm(omega[t_index - 1])**2 * np.eye(3)
        sn = np.linalg.norm(sigmaBR[t_index - 1])
        if (omega_ref is not None):
            term_1= - InertiaTensor @ ( P@(omega[t_index - 1]-omega_ref[t_index-1]) + ( WW + (4/(1+sn**2)* K-0.5*W2) )@sigmaBR[t_index - 1] )
            term_3 = InertiaTensor@(omega_ref_dot_BR-np.cross(omega[t_index-1],omega_ref_BR))
        else:   
            term_1= - InertiaTensor @ ( P@omega[t_index - 1] + ( WW + (4/(1+sn**2)* K-0.5*W2) )@sigmaBR[t_index - 1] )
            term_3 = 0

        term_4 = tilde(omega[t_index - 1]) @ InertiaTensor @ omega[t_index - 1]

        term_5 = -L


        u[t_index-1]= term_1 + term_3 + term_4 + term_5

        if (u[t_index-1][0]>umax):
            u[t_index-1][0]=umax
        if (u[t_index-1][1]>umax):  
            u[t_index-1][1]=umax
        if (u[t_index-1][2]>umax):
            
            u[t_index-1][2]=umax
        if (u[t_index-1][0]<-umax):
            u[t_index-1][0]=-umax
        if (u[t_index-1][1]<-umax):
            u[t_index-1][1]=-umax
        if (u[t_index-1][2]<-umax):
            u[t_index-1][2]=-umax   


        k1s,k1o  = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1],omega[t_index-1],u[t_index-1],L)

        k2s,k2o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1]+0.5*dt*k1s,
                                                              omega[t_index-1]+0.5*dt*k1o,u[t_index-1],L)
        k3s,k3o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                                sigma[t_index-1]+0.5*dt*k2s,
                                                                omega[t_index-1]+0.5*dt*k2o,u[t_index-1],L)    
        k4s,k4o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,     
                                                                sigma[t_index-1]+dt*k3s,
                                                                omega[t_index-1]+dt*k3o,u[t_index-1],L)
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        sigma[t_index] = sigma[t_index - 1] + dt * deltasigma
        omega[t_index] = omega[t_index-1] + dt * deltaomega


        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])
        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])

    return sigma,omega,angles,sigmaBR,u

def EOM_MRP_Control_Integrator(InertiaTensor,sigma0, omega0, t_eval, L = np.array([0,0,0]), K=0, P = np.array([0,0,0]), ffwd=0, fbdklin=0,
                               loadcomp=0, umax = 0, sigma_ref=None,omega_ref=None):
    N=len(t_eval)
    invInertiaTensor = np.linalg.inv(InertiaTensor)
    sigma = np.zeros((N, sigma0.size))
    omega = np.zeros((N, sigma0.size))
    angles = np.zeros((N, sigma0.size))
    sigmaBR = np.zeros((N, sigma0.size))
    H = np.zeros((N, sigma0.size))
    T = np.zeros((N, 1))
    u = np.zeros((N, sigma0.size))
    sigma[0] = sigma0
    omega[0] = omega0
    H[0] = InertiaTensor @ omega0
    T[0] = 0.5*omega0.T @ InertiaTensor @ omega0*0
    angles[0]=MRP2EU_ZYX(sigma0)
    if (sigma_ref is None):
        sigma_ref = np.zeros((N, 3))
        omega_ref = np.zeros((N, 3))
    
    sigmaBR[0] = MRPSum(sigma0,-sigma_ref[0])
        
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]

        #if np.linalg.norm(sigma[t_index-1]) > 1:
        #    sigma[t_index-1] = MRP2Shadow(sigma[t_index-1])
        #if np.linalg.norm(sigma_ref[t_index-1]) > 1:
        #    sigma_ref[t_index-1] = MRP2Shadow(sigma_ref[t_index-1])
        sigmaBR[t_index - 1] = MRPSum(-sigma_ref[t_index - 1],sigma[t_index - 1],)
        if np.linalg.norm(sigmaBR[t_index-1]) >= 1:
            sigmaBR[t_index-1] = MRP2Shadow(sigmaBR[t_index-1])

        BR = MRP2DCM(sigmaBR[t_index - 1])
        omega_ref_BR = BR @ omega_ref[t_index - 1] 
        omega_ref_dot_BR = BR @ ((omega_ref[t_index]-omega_ref[t_index-1])/dt)

        term_1_K= - K * (sigmaBR[t_index - 1]) 

        term_2_P = - P * (omega[t_index - 1] - omega_ref_BR)

        term_3_FF = InertiaTensor@(omega_ref_dot_BR-np.cross(omega[t_index-1],omega_ref_BR))

        term_4_FL = tilde(omega[t_index - 1]) @ InertiaTensor @ omega[t_index - 1]

        term_5_LC = -L

        u[t_index-1]= term_1_K + term_2_P + term_3_FF*ffwd + term_4_FL*fbdklin + term_5_LC*loadcomp

        if (u[t_index-1][0]>umax):
            u[t_index-1][0]=umax
        if (u[t_index-1][1]>umax):  
            u[t_index-1][1]=umax
        if (u[t_index-1][2]>umax):
            u[t_index-1][2]=umax

        if (u[t_index-1][0]<-umax):
            u[t_index-1][0]=-umax
        if (u[t_index-1][1]<-umax):
            u[t_index-1][1]=-umax
        if (u[t_index-1][2]<-umax):
            u[t_index-1][2]=-umax   


        k1s,k1o  = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1],omega[t_index-1],u[t_index-1],L)

        k2s,k2o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                              sigma[t_index-1]+0.5*dt*k1s,
                                                              omega[t_index-1]+0.5*dt*k1o,u[t_index-1],L)
        k3s,k3o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,
                                                                sigma[t_index-1]+0.5*dt*k2s,
                                                                omega[t_index-1]+0.5*dt*k2o,u[t_index-1],L)    
        k4s,k4o = EOM_Rotation_MRP_Differential(InertiaTensor,invInertiaTensor,     
                                                                sigma[t_index-1]+dt*k3s,
                                                                omega[t_index-1]+dt*k3o,u[t_index-1],L)
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        sigma[t_index] = sigma[t_index - 1] + dt * deltasigma
        omega[t_index] = omega[t_index-1] + dt * deltaomega
        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])
        H[t_index] = InertiaTensor @ omega[t_index]
        T[t_index] = 0.5*omega[t_index].T @ InertiaTensor @ omega[t_index]
    return sigma,omega,angles,sigmaBR,u, H, T
