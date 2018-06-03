import numpy as np

X = np.random.rand(10, 10000) * 10
weights = [[2.5, -3.6, -1.23, 3.4, -2.33, 8.67, 1.06, 6.11, -4.23, -9.14]]
weights = np.array(weights)

ans = np.matmul(weights, X)
Y = []
for x_i in ans[0]:
    Y.append(int(x_i > 0))

Y = np.array([Y])

data = np.concatenate((X, Y), axis=0).T

np.savetxt("raw_data.csv", data, delimiter=",")