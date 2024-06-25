import numpy as np
from matplotlib import pyplot as plt
import copy


def multistep(alpha,beta,T,N,yi,f,df):
    y = np.zeros((N + 1, 1))
    k = len(beta) - 1
    y[0:len(yi)] = yi
    h = T/N     #time step
    for l in range(k - 1, N):
        sm = h * (beta[0:k] @ np.array([f(h*(l + 1 + s - k), y[l + 1 + s - k]) for s in range(k)])) - alpha[0:k] @ np.array([y[l+1-k+s] for s in range(k)])

        if beta[-1] != 0:        #implicit case
            def fun(x):
                return sm + h * beta[-1] * f(h * (l+1), x) - x

            def dfun(x):
                return h * beta[-1] * df(h * (l+1), x) - 1

            y_new = Newton(fun, dfun, copy.copy(y[l]))

        else:       #explicit case
            y_new = sm

        y[l+1] = y_new
    return y


def Newton(f, df, x_0):
    tol = 1e-12
    max_steps = 100
    steps = 0
    x = x_0
    while(np.linalg.norm(f(x)) > tol):
        y = np.linalg.solve(df(x), f(x))
        x = x - y
        steps = steps + 1
        if steps > max_steps:
            print("Newton: too many steps")
            break
    return x


def Adams_Moulton2 (N, yi):
    alpha = np.array([0, -1, 1])
    beta = np.array([-1/12, 8/12, 5/12])
    y = multistep(alpha, beta, 1, N, yi, f, df)
    return y


def Adams_Bashforth2 (N, yi):
    alpha = np.array([0, 0, -1, 1])
    beta = np.array([5/12, -16/12, 23/12, 0])
    y = multistep(alpha, beta, 1, N, yi, f, df)
    return y


def Adams_Bashforth1 (N, yi):
    alpha = np.array([0, -1, 1])
    beta = np.array([-1/2, 3/2, 0])
    y = multistep(alpha, beta, 1, N, yi, f, df)
    return y



def f(t, x):
    return x

def df(t, x):
    return np.array([[1]])


def solve(N):
    mesh = np.array([j / N for j in range(N + 1)])
    plt.plot(mesh, Adams_Bashforth1(N, np.array([[np.exp(0)], [np.exp(1 / N)]])), label="Adams-Bashforth k = 1")
    plt.plot(mesh, Adams_Bashforth2(N, np.array([[np.exp(0)], [np.exp(1/N)], [np.exp(2/N)]])), label="Adams-Bashforth k = 2")
    plt.plot(mesh, Adams_Moulton2(N, np.array([[np.exp(0)], [np.exp(1/N)]])), label="Adams-Moulton k = 2")
    plt.plot(mesh, np.exp(mesh), label="true sol")
    plt.legend()
    plt.show()



def error_exact_start():
    e = np.exp(1)
    N = [4, 8, 16, 32, 64, 128, 256]
    h = list(map(lambda x : 1/x, N))
    error_AB1, error_AB2, error_AM2 = np.zeros(len(N)), np.zeros(len(N)), np.zeros(len(N))
    for i in range(len(N)):
        error_AB1[i] = np.abs(e - Adams_Bashforth1(N[i], np.array([[np.exp(0)], [np.exp(1/N[i])], [np.exp(2/N[i])]]))[-1][0])
        error_AB2[i] = np.abs(e - Adams_Bashforth2(N[i], np.array([[np.exp(0)], [np.exp(1/N[i])], [np.exp(2/N[i])]]))[-1][0])
        error_AM2[i] = np.abs(e - Adams_Moulton2(N[i], np.array([[np.exp(0)], [np.exp(1/N[i])]]))[-1][0])


    plt.loglog(h, error_AB1, label = "error AB k = 1", color="red")
    plt.loglog(h, error_AB2, label="error AB k = 2", color="blue")
    plt.loglog(h, error_AM2, label="error AM k = 2", color="green")
    plt.loglog(h, list(map(lambda x: x**2, h)), label="$f(h) = h^2$", linewidth=0.5, color="black")
    plt.loglog(h, list(map(lambda x: x**3, h)), label="$f(h) = h^3$", linewidth=0.5, color="black")
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("error at t = 1")
    plt.show()


def explEuler (f, mesh, y_0):
    y = np.array([y_0])
    n = len(mesh)
    for i in range(n-1):
        h = mesh[i+1] - mesh[i]     # step size
        y_new = y[i] + h * f(mesh[i], y[i])     # compute y_(i+1)
        y = np.append(y, np.array([y_new]), axis=0)
    return y


def error_Euler_start():
    e = np.exp(1)
    N = [4, 8, 16, 32, 64, 128, 256]
    h = list(map(lambda x: 1 / x, N))
    error_AB1, error_AB2 = np.zeros(len(N)), np.zeros(len(N))
    for i in range(len(N)):
        yi_h_AB1 = explEuler(f, np.array([0, 1 / N[i]]), np.array([1]))
        yi_h_AB2 = explEuler(f, np.array([0, 1 / N[i], 2 / N[i]]), np.array([1]))
        error_AB1[i] = np.abs(e - Adams_Bashforth1(N[i], yi_h_AB1)[-1][0])
        error_AB2[i] = np.abs(e - Adams_Bashforth2(N[i], yi_h_AB2)[-1][0])
    plt.loglog(h, error_AB1, label="error AB1 + Euler h = 1/N")
    plt.loglog(h, error_AB2, label="error AB2 + Euler h = 1/N")

    error_AB1, error_AB2 = np.zeros(len(N)), np.zeros(len(N))
    for i in range(len(N)):
        init = explEuler(f, np.linspace(0, 1/N[i], num=(N[i] + 1), endpoint=True), np.array([1]))
        yi_h_AB1 = np.array([init[0], init[N[i]]])
        init = explEuler(f, np.linspace(0, 2/N[i], num=(2 * N[i] + 1), endpoint=True), np.array([1]))
        yi_h_AB2 = np.array([init[0], init[N[i]], init[2*N[i]]])
        error_AB1[i] = np.abs(e - Adams_Bashforth1(N[i], yi_h_AB1)[-1][0])
        error_AB2[i] = np.abs(e - Adams_Bashforth2(N[i], yi_h_AB2)[-1][0])
    plt.loglog(h, error_AB1, label="error AB1 + Euler $h = 1/N^2$")
    plt.loglog(h, error_AB2, label="error AB2 + Euler $h = 1/N^2$")

    plt.loglog(h, list(map(lambda x: x**2, h)), label="$f(h) = h^2$", linewidth=0.5, color="black")
    plt.loglog(h, list(map(lambda x: x**3, h)), label="$f(h) = h^3$", linewidth=0.5, color="black")
    plt.xlabel("h")
    plt.ylabel("error at t = 1")
    plt.legend()
    plt.show()




solve(20)
error_exact_start()
error_Euler_start()

