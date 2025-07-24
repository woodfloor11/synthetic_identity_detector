import numpy as np
y = np.load("data/y.npy")
print("Real:", np.sum(y == 1))
print("Fake:", np.sum(y == 0))
 