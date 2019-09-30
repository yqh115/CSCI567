import numpy as np

R = [1, 0, 1]
print(np.identity(3))
print("/n")
print(np.identity(3)[R])
print("/n")
print(np.sum(np.identity(3)[R], axis=0))