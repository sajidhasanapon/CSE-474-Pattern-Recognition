import numpy as np

L = []
for i in range(100):
    x = np.random.rand()
    L.append(x < 0.5)

L = map(int, L)
for n in L:
    print(n, end="")