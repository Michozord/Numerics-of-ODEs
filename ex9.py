import numpy as np
from matplotlib import pyplot as plt


def explEuler(f, mesh, y_0):       # p = 1
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n-1):
        h = mesh[i+1] - mesh[i]     # step size
        y_new = y[i] + h * f(mesh[i], y[i])     # compute y_(i+1)
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def Simpson(f, mesh, y_0):      # p = 4
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n - 1):
        h = mesh[i + 1] - mesh[i]  # step size
        k = [0, 0, 0, 0]
        k[0] = f(mesh[i], y[i])
        k[1] = 2 * f(mesh[i] + h / 2, y[i] + h / 2 * k[0])
        k[2] = 2 * f(mesh[i] + h / 2, y[i] + h / 2 * k[1])
        k[3] = f(mesh[i] + h, y[i] + h * k[2])
        increment = sum(k) / 6
        y_new = y[i] + h * increment
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def f(t, y):
    v = np.abs(1 - y) + 1
    return v


def solve():
    mesh = np.linspace(0, 5, 50000)
    y_0 = 2
    y_euler = explEuler(f, mesh, y_0)
    y_simpson = Simpson(f, mesh, y_0)
    plt.plot(mesh, y_euler, color = 'blue', label = "Expl. Euler")
    plt.plot(mesh, y_simpson, color = 'red', label = "Simpson")
    plt.legend()
    plt.title("Solution for y_0 = 2")
    plt.show()


def error():
    y_0 = 2
    a, b = 0, 5
    true_solution = Simpson(f, np.linspace(a, b, 100000), y_0)
    error_Euler = []
    error_Simpson = []
    n = [32 * (5**j) for j in range(5)]
    for i in n:
        print("n = ", i)
        mesh = np.linspace(a, b, i)
        sol = explEuler(f, mesh, y_0)
        error_Euler.append(np.abs(sol[i - 1] - true_solution[100000 - 1]))
        print("Error Euler = ", error_Euler[-1])
        sol = Simpson(f, mesh, y_0)
        error_Simpson.append(np.abs(sol[i - 1] - true_solution[100000 - 1]))
        print("Error Simpson = ", error_Simpson[-1])
    plt.loglog(n, error_Euler, color = 'blue', label = 'Error expl. Euler (p = 1)')
    plt.loglog(n, error_Simpson, color='red', label='Error Simpson (p = 4)')
    plt.xlabel("Number of steps")
    plt.ylabel("Error at t = 5")
    plt.title("y_0 = 2")
    plt.legend()
    plt.show()



error()
#solve()



