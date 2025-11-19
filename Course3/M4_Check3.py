import numpy as np

g1 = np.array([0.267261,0.534522,0.801784])
g2 = np.array([-0.267261,0.534522,0.801784])
g3 = np.array([0.534522,0.267261,0.801784])
g4 = np.array([-0.666667,0.666667,0.333333])

G = np.column_stack([g1, g2, g3, g4])
print(G.shape)  # (3, 4)
print(G)

L = np.array([0.1,0.2,0.4])

G2i = np.linalg.inv(G @ G.T)
U = G.T @ G2i @ L
print(G2i)

print(U)