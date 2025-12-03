import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *



def EOM_VSCMG_Single(IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs, gt, gg, L, us=0, ug=0):

    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v

    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])
    InertiaTensor_J_B = Js*gs.T@gs + Jt*gt.T@gt + Jg*gg.T@gg 
    InertiaTensor_RJ_B = InertiaTensor_S_B + (Js-IWs)*gs.T@gs + Jt*gt.T@gt
    InertiaTensor_B = InertiaTensor_S_B + InertiaTensor_J_B

    ws = gs.T @ omega
    wt = gt.T @ omega
    wg = gg.T @ omega

    ugs = gs * (us -IWs*wt*gamma_dot + gamma_dot*wt*(Js-Jt+Jg))
    ugt = gt * (gamma_dot*ws*(Js-Jt-Jg) + IWs*bigOmega*wg)
    ugg = gg * (ug + ws*wt*(Js-Jt))

    w_tilde = tilde(omega)

    inv_InertiaTensor_RJ_B= np.linalg.inv(InertiaTensor_RJ_B)

    X = -w_tilde @ InertiaTensor_B @ omega -ugs - ugt - ugg + L

    omega_dot = inv_InertiaTensor_RJ_B @ ( X )

    bigOmega_dot = us/IWs - gamma_dot*wt - gs.T@omega_dot
    gamma_dot_dot = 1/Jg*(ug + ws*wt*(Js-Jt)+IWs*bigOmega*wt) - gg.T@omega_dot

    return omega_dot, gamma_dot_dot, bigOmega_dot

def EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, sigma, omega,
                                                   gamma, gamma_dot, bigOmega,gs,gt,gg,L):

    sigma_dot = MRP_Differential(sigma, omega)
    omega_dot, gamma_dot_dot, bigOmega_dot = EOM_VSCMG_Single(IS_v,IJ_v, IWs, omega, gamma, gamma_dot, 
                     bigOmega, gs, gt, gg, L)

    delta_sigma = sigma_dot*dt
    delta_omega = omega_dot*dt
    deta_gamma_dot = gamma_dot_dot*dt
    delta_gamma = deta_gamma_dot*dt
    delta_bigOmega = bigOmega_dot*dt

    return delta_sigma, delta_omega, delta_gamma, deta_gamma_dot, delta_bigOmega


def EOM_MRP_VSCMG_Single_Integrator(IS_v,IJ_v,IWs,sigma0, omega0, t_eval, gs0, gt0, gg0, 
                                    gamma0, gamma_dot0, bigOmega0, L):

    # Simulation time lenght
    N=len(t_eval)
    
    # Inertia elements assignement
    Is1,Is2,Is3 = IS_v
    Js,Jt,Jg = IJ_v

    InertiaTensor_S_B = np.array([[Is1,0,0],[0,Is2,0],[0,0,Is3]])
    InertiaTensor_J_G = np.array([[Js,0,0],[0,Jt,0],[0,0,Jg]])

    # empty output array definition
    sigma = np.zeros((N, 3))
    omega = np.zeros((N, 3))
    gamma = np.zeros((N, 1))
    gamma_dot = np.zeros((N, 1))
    bigOmega = np.zeros((N, 1))
    angles = np.zeros((N, 3))
    H_B = np.zeros((N, 3))
    T = np.zeros((N, 1))

    # Initialization
    sigma[0] = sigma0
    omega[0] = omega0
    gamma[0] = gamma0
    gamma_dot[0] = gamma_dot0
    bigOmega[0] = bigOmega0

    gs = gs0
    gt = gg0
    gg = gg0

    H_B[0] = 0
    T[0] = 0

    angles[0]=MRP2EU_ZYX(sigma0)
    
    for t_index in range(1, len(t_eval)):
        dt = t_eval[t_index] - t_eval[t_index - 1]
        s = sigma[t_index-1]
        o = omega[t_index-1]
        g = gamma[t_index-1]
        gdot = gamma_dot[t_index-1]
        bo = bigOmega[t_index-1]

        gs = gs0*np.cos(g-gamma0) + gt0*np.sin(g-gamma0)
        gt = -gs0*np.sin(g-gamma0) + gt0*np.cos(g-gamma0)
        gg = gg0


        k1s, k1o, k1g, k1gdot, k1bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s, o,
                                                   g, gdot, bo, gs,gt,gg,L)

        k2s, k2o, k2g, k2gdot, k2bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+0.5*k1s, 
                                                                          o+0.5*k1o, g+0.5*k1g, gdot+0.5*k1gdot,
                                                                          bo+0.5*k1bo, gs,gt,gg,L)
        
        k3s, k3o, k3g, k3gdot, k3bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+0.5*k2s, 
                                                                          o+0.5*k2o, g+0.5*k2g, gdot+0.5*k2gdot,
                                                                          bo+0.5*k2bo, gs,gt,gg,L)

        k4s, k4o, k4g, k4gdot, k4bo  = EOM_MRP_VSCMG_Single_Differential(dt, IS_v,IJ_v, IWs, s+k3s, 
                                                                          o+k3o, g+k3g, gdot+k3gdot,
                                                                          bo+k3bo, gs,gt,gg,L)
        # body frame angular velocity and MRP update
        deltasigma = (1/6)*(k1s + 2*k2s + 2*k3s + k4s)
        deltaomega = (1/6)*(k1o + 2*k2o +  2*k3o + k4o)
        deltagammadot = (1/6)*(k1gdot + 2*k2gdot +  2*k3gdot + k4gdot)
        deltagamma = (1/6)*(k1g + 2*k2g +  2*k3g + k4g)
        deltabigOmega = (1/6)*(k1bo + 2*k2bo +  2*k3bo + k4bo)

        sigma[t_index] = sigma[t_index - 1]  + deltasigma
        omega[t_index] = omega[t_index-1] + deltaomega
        gamma_dot[t_index] = gamma_dot[t_index-1] + deltagammadot
        gamma[t_index] = gamma[t_index-1] + deltagamma
        bigOmega[t_index] = bigOmega[t_index-1] + deltabigOmega

        if np.linalg.norm(sigma[t_index]) > 1:
            sigma[t_index] = MRP2Shadow(sigma[t_index])        
        angles[t_index]=MRP2EU_ZYX(sigma[t_index])


        H_B[t_index] = 0
        T[t_index] = 0

    return sigma,omega,angles,gamma_dot,gamma,bigOmega,H_B,T
