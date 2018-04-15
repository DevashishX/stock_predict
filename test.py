import numpy as np
x = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
y = np.eye(3, 1)
print(x[:, 0])
print(y)
z = np.c_[y, x[:, 0]]
print(z)
q = np.array([])
q = np.insert(q, 0, x[:, 0], axis = 1)
print(q)
