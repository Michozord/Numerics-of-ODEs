import numpy as np
from matplotlib import pyplot as plt

k_0 = 1


def fdm(N):
    mesh = np.linspace(0, 2*np.pi, num=N+1)
    h = 2*np.pi / N
    y = np.zeros(N+1)
    y[0], y[-1] = 1, 1
    y_unknowns = np.zeros(N-1)
    rhs = np.array(list(map(np.cos, mesh[1:-1])))
    rhs[0] = rhs[0] + k_0/(h*h) * y[0]
    rhs[-1] = rhs[-1] + k_0/(h*h) * y[-1]
    A = 1/(h*h) * (2 * np.eye(N-1) - np.eye(N-1, k=1) - np.eye(N-1, k=-1))
    y_unknowns = np.linalg.solve(A, rhs)
    y[1:-1] = y_unknowns
    return y


def solution():
    N = 40
    mesh = np.linspace(0, 2*np.pi, num=N+1)
    y = fdm(N)
    true_sol = np.array(list(map(lambda x: 1/k_0 * np.cos(x) + 1 - 1/k_0, mesh)))
    plt.plot(mesh, y, label="FDM, N=40")
    plt.plot(mesh, true_sol, label="true solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.show()


def error():
    ns = [5, 10, 20, 80, 100, 1000]
    errors = []
    for N in ns:
        mesh = np.linspace(0, 2 * np.pi, num=N + 1)
        y = fdm(N)
        error = y - np.array(list(map(lambda x: 1 / k_0 * np.cos(x) + 1 - 1 / k_0, mesh)))
        errors = errors + [np.linalg.norm(error, ord = np.inf)]

    plt.loglog(ns, errors, label="max. error")
    plt.loglog(ns, list(map(lambda x: x**(-2), ns)), linewidth=0.7, color="black", label="$N^{-2}$")
    plt.legend()
    plt.ylabel("error")
    plt.xlabel("N")
    plt.show()


solution()
error()
