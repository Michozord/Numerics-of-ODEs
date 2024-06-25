import numpy as np
import scipy.integrate as integrate


def Lagrange(c, j):     # returns j-th Lagrange polynomial for points in c
    def inner(t):
        prod = 1
        for i in range(len(c)):
            if j != i:
                prod = prod * (t - c[i]) / (c[j] - c[i])
        return prod
    return inner


def RK(N, type="closed"):
    c = np.zeros(N+1)
    if type == "closed":
        c = np.array([l/N for l in range(N+1)])     # c_l = l/N for l = 0, ..., N
    elif type == "open":
        c = np.array([(l + 1) / (N + 2) for l in range(N + 1)])  # c_l = l+1/N+2 for l = 0, ..., N
    b = np.array([integrate.quad(Lagrange(c, j), 0, 1)[0] for j in range(N+1)])      # b_j = int from 0 to 1 : L_j(t) dt
    A = np.zeros((N + 1, N + 1))
    for i in range(N + 1):      # A_ij = int from 0 to c_i : L_j(t) dt
        for j in range(N + 1):
            A[i][j] = integrate.quad(Lagrange(c, j), 0, c[i])[0]
    return A, b, c


A, b, c = RK(1, "open")
print(c)
print(A)
print(b)

A, b, c = RK(1, "closed")
print(c)
print(A)
print(b)