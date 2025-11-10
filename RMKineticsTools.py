import numpy as np

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



    