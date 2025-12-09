import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *

def EOM_VSCMG_Single(IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs0, gt0, gg0, gamma0, L, us=0, ug=0):

    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v

    g = gamma
    gs = gs0*np.cos(g-gamma0) + gt0*np.sin(g-gamma0)
    gt = -gs0*np.sin(g-gamma0) + gt0*np.cos(g-gamma0)
    gg = gg0

    ws = np.dot(gs,omega)
    wt = np.dot(gt,omega)
    wg = np.dot(gg,omega)


    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])
    InertiaTensor_J_B = Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gt) 
    InertiaTensor_RJ_B = InertiaTensor_S_B + (Js-IWs)*np.outer(gs,gs) + Jt*np.outer(gt,gt)
    InertiaTensor_B = InertiaTensor_S_B + InertiaTensor_J_B

    ugs = gs * (us -IWs*wt*gamma_dot + gamma_dot*wt*(Js-Jt+Jg))
    ugt = gt * (gamma_dot*ws*(Js-Jt-Jg) + IWs*bigOmega*(wg+gamma_dot))
    ugg = gg * (ug + ws*wt*(Js-Jt))

    w_tilde = tilde(omega)

    inv_InertiaTensor_RJ_B= np.linalg.inv(InertiaTensor_RJ_B)

    X = -(w_tilde @ InertiaTensor_B @ omega) -ugs - ugt - ugg + L

    omega_dot = inv_InertiaTensor_RJ_B @ ( X )
    bigOmega_dot = us/IWs - gamma_dot*wt - np.dot(gs,omega_dot)
    gamma_dot_dot = 1/Jg*(ug + ws*wt*(Js-Jt)+IWs*bigOmega*wt) - np.dot(gg,omega_dot)


    return omega_dot, gamma_dot_dot, bigOmega_dot

def EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, sigma, omega,
                                                   gamma, gamma_dot, bigOmega,gs0,gt0,gg0,gamma0,L):

    sigma_dot = MRP_Differential(sigma, omega)
    delta_sigma = sigma_dot*dt
    delta_gamma = gamma_dot*dt

    omega_dot, gamma_dot_dot, bigOmega_dot = EOM_VSCMG_Single(IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs0, gt0, gg0, gamma0, L)

    delta_omega = omega_dot*dt
    delta_gamma_dot = gamma_dot_dot*dt
    delta_bigOmega = bigOmega_dot*dt

    return delta_sigma, delta_omega, delta_gamma, delta_gamma_dot, delta_bigOmega


def EOM_MRP_VSCMG_Single_Integrator(IS_v,IJ_v,IWs,sigma0, omega0, t_eval, gs0, gt0, gg0, 
                                    gamma0, gamma_dot0, bigOmega0, L):

    # Simulation time lenght
    N=len(t_eval)
    
    # Inertia elements assignement
    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v

    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])

    # empty output array definition
    sigma = np.zeros((N, 3))
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    gamma = np.zeros((N, 1))
    gamma_dot = np.zeros((N, 1))
    bigOmega = np.zeros((N, 1))
    angles = np.zeros((N, 3))
    H_B = np.zeros((N, 3))
    T = np.zeros((N, 1))

    # states Initialization
    sigma[0] = sigma0
    omega[0] = omega0
    gamma[0] = gamma0
    gamma_dot[0] = gamma_dot0
    bigOmega[0] = bigOmega0

    gs = gs0
    gt = gt0
    gg = gg0

    BN = MRP2DCM(sigma[0])
    omega_G_B = omega[0] + gamma_dot[0]*gg
    omega_R_B = omega[0] + gamma_dot[0]*gg + bigOmega[0]*gs
    
    InertiaTensor_J_B = Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gg) 
    InertiaTensor_R_B = IWs*np.outer(gs,gs) 

    HS_B = InertiaTensor_S_B @ omega[0]
    HJ_B = InertiaTensor_J_B @ omega_G_B
    HW_B = InertiaTensor_R_B @ omega_R_B

    H_B[0] = BN.T@(HS_B + HJ_B + HW_B)

    T_B = 0.5*np.dot(omega[0], HS_B)
    T_G = 0.5*np.dot(omega_G_B, HJ_B)
    T_R = 0.5*np.dot(omega_R_B, HW_B)

    T[0] = T_B + T_G + T_R


    angles[0]=MRP2EU_ZYX(sigma0)
    
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        s = sigma[t_index-1]
        o = omega[t_index-1]
        g = gamma[t_index-1]
        gdot = gamma_dot[t_index-1]
        bo = bigOmega[t_index-1]


        k1s, k1o, k1g, k1gdot, k1bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s, o,
                                                   g, gdot, bo, gs0,gt0,gg0,gamma0,L)

        k2s, k2o, k2g, k2gdot, k2bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+0.5*k1s, 
                                                                          o+0.5*k1o, g+0.5*k1g, gdot+0.5*k1gdot,
                                                                          bo+0.5*k1bo, gs0,gt0,gg0,gamma0,L)
        
        k3s, k3o, k3g, k3gdot, k3bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+0.5*k2s, 
                                                                          o+0.5*k2o, g+0.5*k2g, gdot+0.5*k2gdot,
                                                                          bo+0.5*k2bo, gs0,gt0,gg0,gamma0,L)

        k4s, k4o, k4g, k4gdot, k4bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+k3s, 
                                                                          o+k3o, g+k3g, gdot+k3gdot,
                                                                          bo+k3bo, gs0,gt0,gg0,gamma0,L)
        # body frame angular velocity and MRP update
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        deltagammadot = (1/6)*(k1gdot + 2*k2gdot +  2*k3gdot + k4gdot)
        deltagamma = (1/6)*(k1g + 2*k2g +  2*k3g + k4g)
        deltabigOmega = (1/6)*(k1bo + 2*k2bo +  2*k3bo + k4bo)

        #deltasigma = k1s
        #deltaomega = k1o
        #deltagammadot = k1gdot
        #deltagamma = k1g
        #deltabigOmega = k1bo

        sigma[t_index] = sigma[t_index - 1]  + deltasigma
        omega[t_index] = omega[t_index-1] + deltaomega
        gamma_dot[t_index] = gamma_dot[t_index-1] + deltagammadot
        gamma[t_index] = gamma[t_index-1] + deltagamma
        bigOmega[t_index] = bigOmega[t_index-1] + deltabigOmega

        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])

        g = gamma[t_index]
        gs = gs0*np.cos(g-gamma0) + gt0*np.sin(g-gamma0)
        gt = -gs0*np.sin(g-gamma0) + gt0*np.cos(g-gamma0)
        gg = gg0

        BN = MRP2DCM(sigma[t_index])
        omega_G_B = omega[t_index] + gamma_dot[t_index]*gg
        omega_R_B = omega[t_index] + gamma_dot[t_index]*gg + bigOmega[t_index]*gs

        InertiaTensor_J_B = Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gg) 
        InertiaTensor_R_B = IWs*np.outer(gs,gs) 

        HS_B = InertiaTensor_S_B @ omega[t_index]
        HJ_B = InertiaTensor_J_B @ omega_G_B
        HW_B = InertiaTensor_R_B @ omega_R_B

        H_B[t_index] = BN.T@(HS_B + HJ_B + HW_B)

        T_B = 0.5*np.dot(omega[t_index], HS_B)
        T_G = 0.5*np.dot(omega_G_B, HJ_B)
        T_R = 0.5*np.dot(omega_R_B, HW_B)

        T[t_index] = T_B + T_G + T_R

    return sigma,omega,angles,gamma_dot,gamma,bigOmega,H_B,T

# Miltiple VSCMGs EOM with identical wheels and gimbals but having different orientations
def EOM_VSCMG_Multi(num_gimb,IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs0, gt0, gg0, gamma0, L, us=None, ug=None):

    if (us is None):
        us = np.zeros(num_gimb)
    if (ug is None):
        ug = np.zeros(num_gimb)

    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v
    
    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])
    InertiaTensor_RJ_B = InertiaTensor_S_B
    X = np.zeros(3)

    for i in range(num_gimb):
        gs = gs0[:,i]*np.cos(gamma[i]-gamma0[i]) + gt0[:,i]*np.sin(gamma[i]-gamma0[i])
        gt = -gs0[:,i]*np.sin(gamma[i]-gamma0[i]) + gt0[:,i]*np.cos(gamma[i]-gamma0[i])
        gg = gg0[:,i]

        ws = np.dot(gs,omega)
        wt = np.dot(gt,omega)
        wg = np.dot(gg,omega)

        InertiaTensor_S_B += Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gg) 
        #InertiaTensor_RJ_B += (Js-IWs)*np.outer(gs,gs) + Jt*np.outer(gt,gt)
        InertiaTensor_RJ_B += (Js-IWs)*np.outer(gs,gs) + Jt*np.outer(gt,gt)

        ugs = gs * (us[i] -IWs*wt*gamma_dot[i] + gamma_dot[i]*wt*(Js-Jt+Jg))
        ugt = gt * (gamma_dot[i]*ws*(Js-Jt-Jg) + IWs*bigOmega[i]*(wg+gamma_dot[i]))
        ugg = gg * (ug[i] + ws*wt*(Js-Jt))

        X += -ugs -ugt -ugg

    inv_InertiaTensor_RJ_B= np.linalg.inv(InertiaTensor_RJ_B)
    #inv_InertiaTensor_RJ_B= np.linalg.inv(InertiaTensor_S_B)
    w_tilde = tilde(omega)
    X += -(w_tilde @ InertiaTensor_S_B @ omega) + L
    omega_dot = inv_InertiaTensor_RJ_B @ ( X )

    bigOmega_dot = np.zeros(num_gimb)
    gamma_dot_dot = np.zeros(num_gimb)
    for i in range(num_gimb):

        gs = gs0[:,i]*np.cos(gamma[i]-gamma0[i]) + gt0[:,i]*np.sin(gamma[i]-gamma0[i])
        gt = -gs0[:,i]*np.sin(gamma[i]-gamma0[i]) + gt0[:,i]*np.cos(gamma[i]-gamma0[i])
        gg = gg0[:,i]

        ws = np.dot(gs,omega)
        wt = np.dot(gt,omega)
        wg = np.dot(gg,omega)

        bigOmega_dot[i] = us[i]/IWs - gamma_dot[i]*wt - np.dot(gs,omega_dot)
        gamma_dot_dot[i] = 1/Jg*(ug[i] + ws*wt*(Js-Jt)+IWs*bigOmega[i]*wt) - np.dot(gg,omega_dot)

    return omega_dot, gamma_dot_dot, bigOmega_dot

def EOM_MRP_VSCMG_Multi_Integrator(num_gimb, IS_v,IJ_v,IWs,sigma0, omega0, t_eval, gs0, gt0, gg0, 
                                    gamma0, gamma_dot0, bigOmega0, L):

    # Simulation time lenght
    N=len(t_eval)
    
    # Inertia elements assignement
    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v

    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])

    # empty output array definition
    sigma = np.zeros((3, N))
    omega = np.zeros((3, N))
    omega_dot = np.zeros((3, N))
    gamma = np.zeros((num_gimb, N))
    gamma_dot = np.zeros((num_gimb, N))
    bigOmega = np.zeros((num_gimb, N))
    angles = np.zeros((3, N))
    H_N = np.zeros((3, N))
    HS_B = np.zeros((3, N))
    HJ_B = np.zeros((3,num_gimb))
    HW_B = np.zeros((3,num_gimb))
    TG = np.zeros((num_gimb))
    TR = np.zeros((num_gimb))
    T = np.zeros((N))

    # states Initializations
    sigma[:,0] = sigma0
    omega[:,0] = omega0
    BN = MRP2DCM(sigma[:,0])

    for i in range(num_gimb):
        gamma[i,0] = gamma0[i]
        gamma_dot[i,0] = gamma_dot0[i]
        bigOmega[i,0] = bigOmega0[i]
        gs = gs0[:,i] 
        gt = gt0[:,i]
        gg = gg0[:,i]
        omega_G_B = omega[:,0] + gamma_dot0[i]*gg
        omega_R_B = omega[:,0] + gamma_dot0[i]*gg + bigOmega0[i]*gs

        InertiaTensor_J_B = Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gg) 
        InertiaTensor_R_B = IWs*np.outer(gs,gs) 
        
        HJ_B[:,i] = InertiaTensor_J_B @ omega_G_B
        HW_B[:,i] = InertiaTensor_R_B @ (omega_R_B)
        TG[i] = 0.5*np.dot(omega_G_B, HJ_B[:,i])
        TR[i] = 0.5*np.dot(omega_R_B, HW_B[:,i])

    
    HS_B[:,0] = InertiaTensor_S_B @ omega[:,0]
    T[0] = 0.5*np.dot(omega[:,0], HS_B[:,0])

    for i in range(num_gimb):
        HS_B[:,0] += HJ_B[:,i] + HW_B[:,i]
        T[0] += TG[i] + TR[i]
    
    H_N[:,0] = BN.T@(HS_B[:,0])
    
    angles[:,0]=MRP2EU_ZYX(sigma0)
    
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        s = sigma[:,t_index-1]
        o = omega[:,t_index-1]
        g = gamma[:,t_index-1]
        gdot = gamma_dot[:,t_index-1]
        bo = bigOmega[:,t_index-1]

        
        k1s, k1o, k1g, k1gdot, k1bo  = EOM_MRP_VSCMG_Multi_Differential(num_gimb, dt, IS_v,IJ_v, IWs, s,
                                                                            o, g, gdot, bo, gs0,gt0,gg0,gamma0,L)
        
        k2s, k2o, k2g, k2gdot, k2bo  = EOM_MRP_VSCMG_Multi_Differential(num_gimb, dt, IS_v,IJ_v, IWs, s+0.5*k1s, 
                                                                          o+0.5*k1o, g+0.5*k1g, gdot+0.5*k1gdot,
                                                                          bo+0.5*k1bo, gs0,gt0,gg0,gamma0,L)
        
        k3s, k3o, k3g, k3gdot, k3bo  = EOM_MRP_VSCMG_Multi_Differential(num_gimb, dt, IS_v,IJ_v, IWs, s+0.5*k2s, 
                                                                          o+0.5*k2o, g+0.5*k2g, gdot+0.5*k2gdot,
                                                                          bo+0.5*k2bo, gs0,gt0,gg0,gamma0,L)

        k4s, k4o, k4g, k4gdot, k4bo  = EOM_MRP_VSCMG_Multi_Differential(num_gimb, dt, IS_v,IJ_v, IWs, s+k3s, 
                                                                          o+k3o, g+k3g, gdot+k3gdot,
                                                                          bo+k3bo, gs0,gt0,gg0,gamma0,L)
    
        
        # body frame angular velocity and MRP update
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        deltagammadot = (1/6)*(k1gdot + 2*k2gdot +  2*k3gdot + k4gdot)
        deltagamma = (1/6)*(k1g + 2*k2g +  2*k3g + k4g)
        deltabigOmega = (1/6)*(k1bo + 2*k2bo +  2*k3bo + k4bo)

        sigma[:,t_index] = sigma[:,t_index - 1]  + deltasigma
        omega[:,t_index] = omega[:,t_index-1] + deltaomega
        gamma_dot[:,t_index] = gamma_dot[:,t_index-1] + deltagammadot
        gamma[:,t_index] = gamma[:,t_index-1] + deltagamma
        bigOmega[:,t_index] = bigOmega[:,t_index-1] + deltabigOmega

        if np.linalg.norm(sigma[:,t_index]) > 1:
            sigma[:,t_index] = MRP2Shadow(sigma[:,t_index])        
        angles[:,t_index]=MRP2EU_ZYX(sigma[:,t_index])


        BN = MRP2DCM(sigma[:,t_index])

        for i in range(num_gimb):   

            g = gamma[i,t_index]
            gs = gs0[:,i]*np.cos(g-gamma0[i]) + gt0[:,i]*np.sin(g-gamma0[i])
            gt = -gs0[:,i]*np.sin(g-gamma0[i]) + gt0[:,i]*np.cos(g-gamma0[i])
            gg = gg0[:,i]

            omega_G_B = omega[:,t_index] + gamma_dot[i,t_index]*gg
            omega_R_B = omega[:,t_index] + gamma_dot[i,t_index]*gg + bigOmega[i,t_index]*gs

            InertiaTensor_J_B = Js*np.outer(gs,gs) + Jt*np.outer(gt,gt) + Jg*np.outer(gg,gg) 
            InertiaTensor_R_B = IWs*np.outer(gs,gs) 
        
            HJ_B[:,i] = InertiaTensor_J_B @ omega_G_B
            HW_B[:,i] = InertiaTensor_R_B @ (omega_R_B)

            ws = np.dot(gs,omega[:,t_index])
            wt = np.dot(gt,omega[:,t_index])
            wg = np.dot(gg,omega[:,t_index])
        
            #TG[i] = 0.5*np.dot(omega_G_B, HJ_B[:,i])
            #TR[i] = 0.5*np.dot(omega_R_B, HW_B[:,i])
            TG[i] = 0.5*((Js-IWs)*ws**2 + Jt*wt**2 + Jg*(wg+gamma_dot[i,t_index])**2)
            TR[i] = 0.5*(IWs*(ws + bigOmega[i,t_index])**2)
    
        HS_B[:,t_index] = InertiaTensor_S_B @ omega[:,t_index]
        #T[t_index] = 0.5*np.dot(omega[:,t_index], HS_B[:,t_index])
        T[t_index] = 0.5*(Is1*omega[0,t_index]**2 + Is2*omega[1,t_index]**2 + Is3*omega[2,t_index]**2)

        for i in range(num_gimb):
            HS_B[:,t_index] += HJ_B[:,i] + HW_B[:,i]
            T[t_index] += TG[i] + TR[i]

        H_N[:,t_index] = BN.T @ (HS_B[:,t_index])

    return sigma,omega,angles,gamma_dot,gamma,bigOmega,H_N,T

def EOM_MRP_VSCMG_Multi_Differential(num_gimb,dt, IS_v,IJ_v, IWs, sigma, omega,
                                                   gamma, gamma_dot, bigOmega,gs0,gt0,gg0,gamma0,L):

    sigma_dot = MRP_Differential(sigma, omega)
    delta_sigma = sigma_dot*dt

    omega_dot, gamma_dot_dot, bigOmega_dot = EOM_VSCMG_Multi(num_gimb,IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs0, gt0, gg0, gamma0, L)

    delta_omega = omega_dot*dt
    delta_gamma_dot = gamma_dot_dot*dt
    delta_bigOmega = bigOmega_dot*dt
    delta_gamma = gamma_dot*dt

    return delta_sigma, delta_omega, delta_gamma, delta_gamma_dot, delta_bigOmega

def EPM_RW_Multi_Differential(num_RW, dt, IRW, IWs, sigma, omega, bigOmega, GS, US, L):
    
    delta_sigma = np.zeros(3)
    delta_omega = np.zeros(3)
    delta_bigOmega = np.zeros(num_RW)

    sigma_dot = MRP_Differential(sigma, omega)
    omega_dot, bigOmega_dot = EOM_RW_Multi(num_RW, IRW, IWs, omega, bigOmega, GS, US , L)

    delta_sigma = sigma_dot*dt
    delta_omega = omega_dot*dt
    delta_bigOmega = bigOmega_dot*dt

    return delta_sigma, delta_omega, delta_bigOmega

def EOM_RW_Multi(num_RW, IRW, IWs, omega, bigOmega, GS, US , L):

    inv_IRW = np.linalg.inv(IRW)
    w_tilde = tilde(omega)
    hs = np.zeros((num_RW))

    for i in range(num_RW):
        gs = GS[:,i]
        ws = np.dot(gs,omega)
        hs[i] = IWs*(ws + bigOmega[i])
        
    X  = -(w_tilde @ IRW @ omega) - w_tilde @ GS @ hs - GS @ US + L
    omega_dot = inv_IRW @ ( X )

    bigOmega_dot = US/IWs - GS.T @ omega_dot

    return omega_dot, bigOmega_dot

def EOM_MRP_RW_Multi_Integrator(num_RW, IS_v, IWs, sigma0, omega0, t_eval, GS, bigOmega0, L):

    #def EOM_RW_Multi(num_RW, IRW, IWs, omega, bigOmega, GS, GT, GG, US , L):

    # Simulation time lenght
    N=len(t_eval)
    
    # Inertia elements assignement
    Is1,Is2,Is3 = IS_v
    # Spacecraft inertia tensor
    IRW = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])

    # empty output array definition
    sigma = np.zeros((3, N))
    omega = np.zeros((3, N))
    omega_dot = np.zeros((3, N))
    bigOmega = np.zeros((num_RW, N))
    angles = np.zeros((3, N))
    H_N = np.zeros((3, N))
    HS_B = np.zeros((3, N))
    TR = np.zeros((num_RW))
    T = np.zeros((N))

    # states Initializations
    sigma[:,0] = sigma0
    omega[:,0] = omega0
    
    BN = MRP2DCM(sigma[:,0])

    om = omega[:,0]
    om = om.reshape(3,1)
    omega_R_B = om + GS @ bigOmega0

    aux = IRW @ om + IWs * omega_R_B
    HS_B[:,0] = aux.T
    T[0] = 0.5*np.dot(omega[:,0], HS_B[:,0]) + 0.5*IWs*np.dot(omega_R_B.T, omega_R_B)

    H_N[:,0] = BN.T@(HS_B[:,0])
    
    angles[:,0]=MRP2EU_ZYX(sigma0)
    
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        s = sigma[:,t_index-1]
        o = omega[:,t_index-1]
        bo = bigOmega[:,t_index-1]

        US = np.array([.1,0.0,0.0,0.0])
        k1s, k1o, k1bo  = EPM_RW_Multi_Differential(num_RW, dt, IRW, IWs, s, o, bo, GS, US, L)
        
        k2s, k2o, k2bo  = EPM_RW_Multi_Differential(num_RW, dt, IRW, IWs, s+0.5*k1s, 
                                                                          o+0.5*k1o, bo+0.5*k1bo, GS, US,L)
        
        k3s, k3o, k3bo  = EPM_RW_Multi_Differential(num_RW, dt, IRW, IWs, s+0.5*k2s, 
                                                                          o+0.5*k2o, bo+0.5*k2bo, GS,US, L)

        k4s, k4o, k4bo  = EPM_RW_Multi_Differential(num_RW, dt, IRW, IWs, s+k3s, 
                                                                          o+k3o, bo+k3bo, GS, US ,L)
        

        # body frame angular velocity and MRP update
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        deltabigOmega = (1/6)*(k1bo + 2*k2bo +  2*k3bo + k4bo)
        
        sigma[:,t_index] = sigma[:,t_index - 1]  + deltasigma
        omega[:,t_index] = omega[:,t_index-1] + deltaomega
        bigOmega[:,t_index] = bigOmega[:,t_index-1] + deltabigOmega

        if np.linalg.norm(sigma[:,t_index]) > 1:
            sigma[:,t_index] = MRP2Shadow(sigma[:,t_index])        
        angles[:,t_index]=MRP2EU_ZYX(sigma[:,t_index])

        omega_R_B = omega[:,t_index] + GS @ bigOmega[:,t_index]
        HS_B[:,t_index] = IRW @ omega[:,t_index] + IWs * omega_R_B

        TR = 0
        for i in range(num_RW):   
            ws = np.dot(GS[:,i],omega[:,t_index])
            TR += 0.5*(IWs*(ws + bigOmega[i,t_index])**2)
    
        HS_B[:,t_index] = IRW @ omega[:,t_index]
        T[t_index] = 0.5*(Is1*omega[0,t_index]**2 + Is2*omega[1,t_index]**2 + Is3*omega[2,t_index]**2) + TR

        BN = MRP2DCM(sigma[:,t_index])

        H_N[:,t_index] = BN.T @ (HS_B[:,t_index])

    return sigma,omega,angles,bigOmega,H_N,T
