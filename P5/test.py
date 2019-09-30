import numpy as np
a = np.array([4, 3, 9])
b = np.array([1, 2, 6])
index1 = np.argsort(a)
index2 = np.argsort(-a)
c = (a-b)**2

print(c)
print(np.mean(c))
