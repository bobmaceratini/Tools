import numpy as np
from RMKinematicTools import *

rLMO = 3796.19  # Radius of Nano Spacecraft Orbit in km
OmLMO = 20.0/180*np.pi  # Longitude of Ascending Node of Nano Spacecraft Orbit in rad
iLMO = 30.0/180*np.pi  # Inclination of Nano Space craft Orbit in rad
wLMO = 0.000884797  # Argument of Periapsis of Nano Spacecraft Orbit in rad/s
tetaLMO_0 = 60.0/180*np.pi  # True Anomaly of Nano Spacecraft at t=0 in rad

rGMO = 20424.2   # Radius of Mother Spacecraft Orbit in km
OmGMO = 0.0/180*np.pi  # Longitude of Ascending Node of Mother Spacecraft Orbit in rad
iGMO = 0.0/180*np.pi  # Inclination of Mother Space craft Orbit in rad
wGMO = 0.0000709003   # Argument of Periapsis of Mother Spacecraft Orbit in rad/s
tetaGMO_0 = 250.0/180*np.pi  # True Anomaly of Mother Spacecraft at t=0 in rad


# Mod 2 Functions
def get_DCM(Om, i, w, time, teta_0):
    teta = w*time + teta_0
    ZXZ = np.array([Om, i , teta])
    DCM = Eu2DCM_ZXZ(ZXZ)
    return DCM

def get_position(r, Om, i, w, time, teta_0):
    DCM = get_DCM(Om, i, w, time, teta_0)
    rH = np.array([r,0,0])
    rN = DCM.T @ rH
    return rN

def get_velocity(r, Om, i, w, time, teta_0):
    DCM = get_DCM(Om, i, w, time, teta_0)
    vH = np.array([0,r*w,0])
    vN = DCM.T @ vH
    return vN

def get_HN(time):
    HN = get_DCM(OmLMO, iLMO, wLMO, time, tetaLMO_0)
    return HN
    
def get_RsN(time):
    RsN = np.array([[0,0,-1],[0,0,1],[0,1,0]])
    return RsN

def get_RnH(time):
    RsH = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    return RsH

def get_wRsN(time):
    w = 0
    return w

def get_wHN(time):
    wHN_H = np.array([0,0,wLMO])
    wHN_N = get_HN(time).T @ wHN_H
    return wHN_N

def get_RnN(time):
    HN = get_HN(time)
    RnH = get_RnH(time)
    RnN = RnH @ HN
    return RnN

def get_wRnN(time):
    wRnN = get_wHN(time)
    return wRnN

def get_rLMO(time):
    rLMO_t = get_position(rLMO, OmLMO, iLMO, wLMO, time, tetaLMO_0)
    return rLMO_t

def get_rGMO(time):
    rGMO_t = get_position(rGMO, OmGMO, iGMO, wGMO, time, tetaGMO_0)
    return rGMO_t   

def get_Delta_rMO(time):
    rLMO_t = get_rLMO(time)
    rGMO_t = get_rGMO(time)
    Delta_rMO = rGMO_t - rLMO_t
    return Delta_rMO

def get_RcN(time):
    Delta_r = get_Delta_rMO(time)
    rc1 = Delta_r / np.linalg.norm(Delta_r)
    rc2 = np.cross(rc1, np.array([0,0,1]))
    rc2 = rc2 / np.linalg.norm(rc2)
    rc3 = np.cross(rc1, rc2)   
    rc3 = rc3 / np.linalg.norm(rc3) 
    RcN = np.column_stack((rc1, rc2, rc3))
    return RcN


# Task 1
t = 450  # time in seconds for Nano Spacecraft
PLMO = get_position(rLMO, OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")
print("Mod 2 Task 1 - Position of Nano Spacecraft at t=450s (km): ", PLMO)

VLMO = get_velocity(rLMO, OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")
print("Mod 2 Task 1 - Velocity of Nano Spacecraft at t=450s (km/s): ", VLMO)

t = 1150  # time in seconds for Mother Spacecraft
PGMO = get_position(rGMO, OmGMO, iGMO, wGMO, t, tetaGMO_0)
print("---------------------------------------------------")
print("Mod 2 Task 1 - Position of Mother Spacecraft at t=1150s (km): ", PGMO)

VGMO = get_velocity(rGMO, OmGMO, iGMO, wGMO, t, tetaGMO_0)
print("---------------------------------------------------")
print("Mod 2 Task 1 - Velocity of Mother Spacecraft at t=1150s (km/s): ", VGMO)

# Task 2
t = 300  # time in seconds
DCM = get_DCM(OmLMO, iLMO, wLMO, t, tetaLMO_0)
print("---------------------------------------------------")    
print("Mod 2 Task 2 - DCM of Nano Spacecraft at t=300s: \n", DCM)

# Task 3

# Task 4    
t = 330  # time in seconds
RnN = get_RnN(t)
print("---------------------------------------------------")
print("Task 4 - Matrix RnN at t=330s : \n", RnN)

wRnN = get_wRnN(t)
print("---------------------------------------------------")
print("Task 4 - RnN angular velocity at t=330s (rad/s): \n", wRnN)

# Task 5
t = 330  # time in seconds
RcN = get_RcN(t)
print("---------------------------------------------------")
print(f"Task 5 - Matrix RcN at t={t}s: \n", RcN)
