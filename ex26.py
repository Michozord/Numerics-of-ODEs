import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import sympy as sp


def equation_f(eps):
    def inner(t, y):
        r = np.array([0, 0])
        r[0] = y[0] + y[1]
        r[1] = (2*y[0] - y[1])/eps
        return r
    return inner


def is_triangle(A):
    m = A.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if A[i, j] != 0:
                return False
    return True


def is_strict_triangle(A):
    m = A.shape[0]
    for i in range(m):
        if A[i, i] != 0:
            return False
    return True


def RK(f, mesh, y_0, A, b, c):
    y = y_0
    s = len(mesh)
    n = 2
    m = A.shape[0]
    if is_triangle(A):
        if is_strict_triangle(A):     #explicit case
            for i in range(1, s):
                h = mesh[i] - mesh[i - 1]
                k = np.zeros((m, n))
                for j in range(m):
                    sm = sum([A[j, l] * k[l] for l in range(j)])
                    k[j] = f(mesh[i - 1] + h * c[j], y + h * sm)
                y = y + h * b @ k
            return y
        else:   #diagonal implicit case
            for i in range(1, s):
                h = mesh[i] - mesh[i - 1]
                k = np.zeros((m, n))
                for j in range(m):
                    sm = sum([A[j,l] * k[l] for l in range(j)])
                    fun = lambda x: f(mesh[i - 1] + c[j] * h, y + h * sm + h * A[j, j] * x) - x
                    k[j] = fsolve(fun, f(mesh[i-1], y), xtol=1e-5)
                y = y + h * b @ k
            return y
    else:   #implicit case
        for i in range(1, s):
            h = mesh[i] - mesh[i - 1]

            def fun(x):
                k = np.reshape(x, (m, n))
                r = np.zeros((m, n))
                for j in range(m):
                    r[j] = f(mesh[i-1] + c[j] * h, y + h * A[j] @ k) - k[j]
                return np.reshape(r, m*n)

            k_0 = np.tile(f(mesh[i-1], y), m)
            k = fsolve(fun, k_0, xtol=1e-5)
            k = np.reshape(k, (m, n))
            y = y + h * b @ k
        return y


def scipy_solve(epsilon):
    t = sp.symbols('t')
    u = sp.Function('u')(t)
    v = sp.Function('v')(t)

    # Define the system of ODEs
    eq1 = sp.Eq(u.diff(t), u + v)
    eq2 = sp.Eq(v.diff(t), 2 / epsilon * u - 1 / epsilon * v)
    system = [eq1, eq2]

    # Solve the system of ODEs
    sol = sp.dsolve(system, [u, v])

    # Extract the values of u and v at t=0.1
    const = sp.solve([sol[0].rhs.subs(t, 0) - 1, sol[1].rhs.subs(t, 0) - 4], sp.symbols('C1 C2'))
    sol = [eq.rhs.subs(const) for eq in sol]

    return np.array((sol[0].subs(t, 0.1), sol[1].subs(t, 0.1)))


def main():
    b_RK4 = np.array([1/6, 1/3, 1/3, 1/6])
    c_RK4 = np.array([0, 1/2, 1/2, 1])
    A_RK4 = np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]])
    b_Euler = np.array([1])
    c_Euler = np.array([0])
    A_Euler = np.array([[0]])
    b_Gauss = np.array([1 / 2, 1 / 2])
    c_Gauss = np.array([1/2 - np.sqrt(3)/6, 1/2 + np.sqrt(3)/6])
    A_Gauss = np.array([[1/4, 1/4 - np.sqrt(3)/6], [1/4 + np.sqrt(3)/6, 1/4]])
    b_Rad = np.array([3/4, 1/4])
    c_Rad = np.array([1/3, 1])
    A_Rad = np.array([[5/12, -1/12], [3/4, 1/4]])
    b_Rad2 = np.array([2/5 - np.sqrt(6)/10, 2/5 + np.sqrt(6)/10, 1])
    c_Rad2 = np.array([4/9 - np.sqrt(6)/36, 4/9 + np.sqrt(6)/36, 1/9])
    A_Rad2 = np.array([[11/45 - 7/360 * np.sqrt(6), 37/225 - 169/1800 * np.sqrt(6), -2/225 + np.sqrt(6)/75], [37/225 + 169/1800 * np.sqrt(6), 11/45 + 7/360*np.sqrt(6), -2/225 - np.sqrt(6)/75], [4/9 - np.sqrt(6)/36, 4/9 + np.sqrt(6)/36, 1/9]])

    epsilons = [0.0001, 0.001, 0.01, 0.1, 1]
    y_0 = np.array([1, 4])
    mesh = np.linspace(0, 0.1, num=500)
    RK4_solutions = np.zeros((len(epsilons),2))
    Radau_solutions = np.zeros((len(epsilons),2))
    Gauss_solutions = np.zeros((len(epsilons),2))
    Radau2_solutions = np.zeros((len(epsilons), 2))
    Euler_solutions = np.zeros((len(epsilons), 2))
    true_solutions = np.zeros((len(epsilons),2))
    for i in range(len(epsilons)):
        eps = epsilons[i]
        f = equation_f(eps)
        true_solutions[i] = scipy_solve(eps)
        Radau2_solutions[i] = np.abs(RK(f, mesh, y_0, A_Rad2, b_Rad2, c_Rad2) - true_solutions[i])
        RK4_solutions[i] = np.abs(RK(f, mesh, y_0, A_RK4, b_RK4, c_RK4) - true_solutions[i])
        Radau_solutions[i] = np.abs(RK(f, mesh, y_0, A_Rad, b_Rad, c_Rad) - true_solutions[i])
        Gauss_solutions[i] = np.abs(RK(f, mesh, y_0, A_Gauss, b_Gauss, c_Gauss) - true_solutions[i])
        Euler_solutions[i] = np.abs(RK(f, mesh, y_0, A_Euler, b_Euler, c_Euler) - true_solutions[i])


    plt.loglog(epsilons, RK4_solutions[:, 0], label="u error for RK4")
    plt.loglog(epsilons, Radau_solutions[:, 0], label="u error for RadauIIA p = 3")
    plt.loglog(epsilons, Radau2_solutions[:, 0], label="u error for RadauIIA p = 5")
    plt.loglog(epsilons, Gauss_solutions[:, 0], label="u error for Gauss")
    plt.loglog(epsilons, Euler_solutions[:, 0], label="u error for Euler")

    plt.title("u errors at t = 0.1")
    plt.xlabel("$\epsilon$")
    plt.ylabel("")
    plt.legend()
    plt.show()

    plt.loglog(epsilons, RK4_solutions[:, 1], label="v error for RK4")
    plt.loglog(epsilons, Radau_solutions[:, 1], label="v error for RadauIIA p = 3")
    plt.loglog(epsilons, Radau2_solutions[:, 1], label="v error for RadauIIA p = 5")
    plt.loglog(epsilons, Gauss_solutions[:, 1], label="v error for Gauss")
    plt.loglog(epsilons, Euler_solutions[:, 1], label="v error for Euler")

    plt.title("v errors at t = 0.1")
    plt.xlabel("$\epsilon$")
    plt.ylabel("error")
    plt.legend()
    plt.show()


main()
