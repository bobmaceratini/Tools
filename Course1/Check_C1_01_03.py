from RMKineticsTools import *

Points = [
    MassPoint(np.array([1,-1,2]),np.array([2,1,1]),1),
    MassPoint(np.array([-1,-3,2]),np.array([0,-1,1]),1),
    MassPoint(np.array([2,-1,-1]),np.array([3,2,-1]),2),
    MassPoint(np.array([3,-1,-2]),np.array([0,0,1]),2)]
    
print("Points")
print(Points)

CDM = CalcCenterOfMass(Points)

print("Center of mass properties:")
print(CDM)

T_tot = CalcKineticEnergy(Points)
T_cdm = CalcKineticEnergy([CDM])
T_cdm = CalcKineticEnergy([CDM])
T_diff = T_tot - T_cdm

print(f"Total Energy in J: {T_tot}")
print(f"Energy of CDM in J: {T_cdm}")
print(f"Rotational and Deformation Energy in J: {T_diff}")


CDM_momentum = CDM.velocity_v*CDM.mass

print(f"CDM momentum: {CDM_momentum}")

P1 =   MassPoint(np.array([0,0,0]),np.array([0,0,0]),0)
H1 = CalcAngularMomentum(Points,P1)

H2 = CalcAngularMomentum(Points,CDM)

print(f"Angular momentum versus origin: {H1}")
print(f"Angular momentum versus CDM: {H2}")
