import numpy as np
from scipy.linalg import lu_factor, lu_solve
from matplotlib import pyplot as plt


def explEuler(f, mesh, y_0):
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n-1):
        h = mesh[i+1] - mesh[i]     # time step size
        y_new = y[i] + h * f(mesh[i], y[i])     # compute y_(i+1)
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def implEuler(h, N, mesh, y_0):
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n - 1):
        tau = mesh[i + 1] - mesh[i]  # time step size
        M = -2 * np.eye(N - 1) + np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)
        M = (-1/(h*h)) * tau * M + np.eye(N - 1)      # y_l = (Id - tau * M) * y_l+1
        lu, piv = lu_factor(M)
        y_new = lu_solve((lu, piv), y[i])
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def heat_eq(h, tau, T, method="explicit"):
    N = int(1 / h)       # number of spacial steps
    time_steps = int(T/tau)     # number of time-steps
    time_mesh = np.linspace(0, T, num=time_steps)
    G = np.array([g(i/N) for i in range(1, N)])
    if method == "explicit":
        y = explEuler(heat_func(h, N), time_mesh, G)
    elif method == "implicit":
        y = implEuler(h, N, time_mesh, G)
    return time_mesh, y


def g(x):
    return np.exp(-30 * ((x - 0.5)**2))


def heat_func(h, N):
    def inner(t, y):
        M = -2 * np.eye(N-1) + np.eye(N-1, k=1) + np.eye(N-1, k=-1)
        return (1/(h*h)) * M @ y
    return inner


def solve(method):
    T = 2
    hs = [2**(-i) for i in [1, 2, 3, 4]]       # tau < h^2 / 2
    if method == "explicit":
        for h in hs:
            #taus = [(h*h)/2, (h*h)/4, (h*h)/8]
            taus = [h*h, (h*h)/2, (h*h)/4, (h*h)/8]
            for tau in taus:
                mesh, y = heat_eq(h, tau, T, "explicit")
                y_norms = np.array([np.linalg.norm(v, ord=np.inf) for v in y])
                lbl = 'h = ' + str(h) + ', tau = ' + str(tau)
                plt.plot(mesh, y_norms, label=lbl)
                plt.title("$||U_n (t)||_{\infty}$, explicit Euler")
                plt.legend()
            plt.show()
    elif method == 'implicit':
        for h in hs:
            taus = [h*h, (h*h)/2, (h*h)/4, (h*h)/8]
            for tau in taus:
                mesh, y = heat_eq(h, tau, T, "implicit")
                y_norms = np.array([np.linalg.norm(v, ord=np.inf) for v in y])
                lbl = 'h = ' + str(h) + ', tau = ' + str(tau)
                plt.plot(mesh, y_norms, label=lbl)
                plt.title("$||U_n (t)||_{\infty}$, implicit Euler")
                plt.legend()
            plt.show()


solve('implicit')
