import numpy as np
from matplotlib import pyplot as plt

def explEuler (f, mesh, y_0):
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n-1):
        h = mesh[i+1] - mesh[i]     # step size
        y_new = y[i] + h * f(mesh[i], y[i])     # compute y_(i+1)
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def f(t, y):
    alpha, beta, gamma, delta = 2, 0.001, 0.001, 1
    preys = alpha * y[0] - beta * y[0] * y[1]       # y_1 function - amount of preys
    predators = gamma * y[0] * y[1] - delta * y[1]  # y_2 function - amount of predators
    return np.array([preys, predators])

def prey_predators(n, p = False):
    preys_start = 300
    predators_start = 150
    mesh = np.linspace(0, 20, num=n)
    y = explEuler(f, mesh, np.array([preys_start, predators_start]))
    if p:
        preys = y[:, 0]
        predators = y[:, 1]
        plt.plot(mesh, preys, color='green', label='preys')
        plt.plot(mesh, predators, color='red', label='predators')
        plt.legend()
        plt.show()
    return y


def prey_predators_error():
    true_solution = prey_predators(100000)
    n = 1000
    for i in range(5):
        y = prey_predators(n)
        error_0 = y[n - 1, 0] - true_solution[99999, 0]
        error_1 = y[n - 1, 1] - true_solution[99999, 1]
        print("n = ", n, "error_0", error_0, "error_1", error_1)
        n = n * 3

prey_predators(10000, True)
#prey_predators_error()



