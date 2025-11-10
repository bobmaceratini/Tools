import numpy as np
from scipy.spatial.transform import Rotation as R

# Angoli in gradi
euler_BN_deg = [10, 20, 30]   # B rispetto a N
euler_RN_deg = [-5, 5, 5]     # R rispetto a N

# Conversione in rotazioni (ZYX = 3-2-1)
R_BN = R.from_euler('zyx', euler_BN_deg, degrees=True).as_matrix()
R_RN = R.from_euler('zyx', euler_RN_deg, degrees=True).as_matrix()

# Calcolo della rotazione relativa B rispetto a R
R_BR = R_BN @ R_RN.T

# Estrazione degli angoli di Eulero (3-2-1) da R_BR
euler_BN = R.from_matrix(R_BN).as_euler('zyx', degrees=True)
euler_RN = R.from_matrix(R_RN).as_euler('zyx', degrees=True)
euler_BR = R.from_matrix(R_BR).as_euler('zyx', degrees=True)

# Stampa dei risultati
print("Angoli di Eulero (3-2-1) di B rispetto a N:")
print(f"Yaw (Z):   {euler_BN[0]:.4f}°")
print(f"Pitch (Y): {euler_BN[1]:.4f}°")
print(f"Roll (X):  {euler_BN[2]:.4f}°")
print(R_BN)
print("Angoli di Eulero (3-2-1) di R rispetto a N:")
print(f"Yaw (Z):   {euler_RN[0]:.4f}°")
print(f"Pitch (Y): {euler_RN[1]:.4f}°")
print(f"Roll (X):  {euler_RN[2]:.4f}°")
print(R_RN)
print("Angoli di Eulero (3-2-1) di B rispetto a R:")
print(f"Yaw (Z):   {euler_BR[0]:.4f}°")
print(f"Pitch (Y): {euler_BR[1]:.4f}°")
print(f"Roll (X):  {euler_BR[2]:.4f}°")
print(R_BR)
