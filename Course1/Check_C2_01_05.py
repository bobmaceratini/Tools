import numpy as np
from RMKinematicTools import *
from RMKineticsTools import *

wN = np.array([0.01,-0.01,0.01])
teta321_v = np.array([-10,10,5])/180*np.pi

Itens = np.array([[10,1,-1],[1,5,1],[-1,1,8]])

BN = Eu2DCM_ZYX(teta321_v)
wB = BN@wN

HB = Itens @ wB
print(f"angular momentum in body frame: {HB}")