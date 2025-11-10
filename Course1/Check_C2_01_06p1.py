from RMKinematicTools import *
from RMKineticsTools import *
from RMAttitudeStaticDeterminationTools import *


IC_B = np.array([[10,1,-1],[1,5,1],[-1,1,8]])
print("Intertia Tensor vs COM, B frame: ")
print(IC_B)

sigma_DB = np.array([0.1,0.2,0.3])
DB = MRP2DCM(sigma_DB)

IC_D = DB @ IC_B @ DB.T
print("Inertia Tensor cs COM, D frame:")
print(IC_D)

eigenvalues, eigenvectors = np.linalg.eig(IC_B)
print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors[:,0])
print(eigenvectors[:,1])
print(eigenvectors[:,2])

print("--------------------------------")

eig_sort = np.sort(eigenvalues)[::-1]
eig_sort_index = np.argsort(eigenvalues)[::-1]

print("autovalori riordinati:")
print(eig_sort)
print("indici dopo il riordino:")
print(eig_sort_index)

f1 = -eigenvectors[:,eig_sort_index[0]]
f1 = f1/np.linalg.norm(f1)
print(f"autovettore f1: {f1}")
f2 = eigenvectors[:,eig_sort_index[1]]
f2 = f2/np.linalg.norm(f2)
print(f"autovettore f2: {f2}")
f3 = eigenvectors[:,eig_sort_index[2]]
f3 = f3/np.linalg.norm(f3)
print(f"autovettore f3: {f3}")

f3chek = np.cross(f1,f2)
print(f"prodotto vettoriale f1 x f2: {f3chek}")
if (np.dot(f3chek,f3)) < 0:
    f3 = -f3
else:
    f3 = f3



C = np.array([f1,f2,f3])

print("matrice C:")
print(C)
print("Verifica ortonormalità matrice C")
print(C @ C.T)

Idiag = C @ IC_B @ C.T
print("Tensore diagnolizzato:")
print(Idiag)

print(f"\nVerifica autovettore 1")
print("Ip @ v =", IC_B @ f1)
print("λ * v =", eig_sort[0]*f1)
print(f"λ = {eig_sort[0]}")
print(f"v = {f1}")
print(f"\nVerifica autovettore 2")
print("Ip @ v =", IC_B @ f2)
print("λ * v =", eig_sort[1]*f2)
print(f"λ = {eig_sort[1]}")
print(f"v = {f2}")
print(f"\nVerifica autovettore 3")
print("Ip @ v =", IC_B @ f3)
print("λ * v =", eig_sort[2]*f3)
print(f"λ = {eig_sort[2]}")
print(f"v = {f3}")
