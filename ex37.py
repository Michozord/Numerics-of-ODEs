import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

m, l, g = 1, 1, 10


def Hamiltonian(v):
    p, q = v[0], v[1]
    return 0.5 * p**2 - g/l * np.cos(q)


def f(t, v):
    p, q = v[0], v[1]
    return np.array([-g/l * np.sin(q), p])



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
    y = np.array([y_0])
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
                    k[j] = f(mesh[i - 1] + h * c[j], y[-1] + h * sm)
                y_new = y[-1] + h * b @ k
                y = np.append(y, np.array([y_new]), axis=0)
            return y
        else:   #diagonal implicit case
            for i in range(1, s):
                h = mesh[i] - mesh[i - 1]
                k = np.zeros((m, n))
                for j in range(m):
                    sm = sum([A[j,l] * k[l] for l in range(j)])
                    fun = lambda x: f(mesh[i - 1] + c[j] * h, y[-1] + h * sm + h * A[j, j] * x) - x
                    k[j] = fsolve(fun, f(mesh[i-1], y[-1]), xtol=1e-5)
                y_new = y[-1] + h * b @ k
                y = np.append(y, np.array([y_new]), axis=0)
            return y



def symplectic_Euler(f, mesh, y_0):
    y = np.array([y_0])
    s = len(mesh)
    for i in range(1, s):
        h = mesh[i] - mesh[i - 1]
        p_prev, q_prev = y[-1, 0], y[-1, 1]
        q_new = q_prev + h * f(0, [p_prev, 0])[1]
        p_new = p_prev + h * f(0, [0, q_new])[0]
        y = np.append(y, np.array([[p_new, q_new]]), axis=0)
    return y



def main():
    b_RK4 = np.array([1/6, 1/3, 1/3, 1/6])
    c_RK4 = np.array([0, 1/2, 1/2, 1])
    A_RK4 = np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]])
    b_imm = np.array([1])
    A_imm = np.array([[1/2]])
    c_imm = np.array([1/2])

    mesh = np.linspace(0, 10, num=51)
    y_0 = np.array([0, 1])
    solution_RK4 = RK(f, mesh, y_0, A_RK4, b_RK4, c_RK4)
    Hamiltonian_RK4 = list(map(Hamiltonian, solution_RK4))
    solution_imm = RK(f, mesh, y_0, A_imm, b_imm, c_imm)
    Hamiltonian_imm = list(map(Hamiltonian, solution_imm))
    solution_sympl = symplectic_Euler(f, mesh, y_0)
    #print(solution_sympl)
    Hamiltonian_sympl = list(map(Hamiltonian, solution_sympl))
    #print(Hamiltonian_sympl)

    plt.plot(mesh, solution_RK4[:,1], label='RK4')
    plt.plot(mesh, solution_imm[:,1], label='impl. midpoint')
    plt.plot(mesh, solution_sympl[:,1], label='sympl. Euler')

    plt.title('Solution')
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.legend()
    plt.show()

    plt.plot(mesh, solution_RK4[:,0], label='RK4')
    plt.plot(mesh, solution_imm[:,0], label='impl. midpoint')
    plt.plot(mesh, solution_sympl[:,0], label='sympl. Euler')

    plt.title('Solution')
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.legend()
    plt.show()

    plt.plot(mesh, Hamiltonian_RK4, label='RK4')
    plt.plot(mesh, Hamiltonian_imm, label='impl. midpoint')
    plt.plot(mesh, Hamiltonian_sympl, label='sympl. Euler')

    plt.title('Hamiltonian')
    plt.xlabel('t')
    plt.ylabel('$H(p(t), q(t))$')
    plt.legend()
    plt.show()

main()