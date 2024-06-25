import numpy as np
from matplotlib import pyplot as plt


def RK4(f, t, y, h):       # order p = 4
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 55, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3186, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656

    c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1

    b = np.array((35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
    k1 = f(t, y)
    k2 = f(t + c2 * h, y + h * a21 * k1)
    k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
    k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    k = np.array((k1, k2, k3, k4, k5, k6))
    return b @ k


def RK5(f, t, y, h):       # order p = 5
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 55, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3186, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    a71, a73, a74, a75, a76 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

    c2, c3, c4, c5, c6, c7 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1

    b = np.array((5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40))
    k1 = f(t, y)
    k2 = f(t + c2 * h, y + h * a21 * k1)
    k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
    k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    k7 = f(t + c7 * h, y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
    k = np.array((k1, k2, k3, k4, k5, k6, k7))
    return b @ k


def ivp_solver(t_start, t_end, y_0, fun, fi_1=RK4, fi_2=RK5, p=4, h=5e-2, tau=1e-2, h_min=1e-2, lamb=2, rho=0.8):     # adaptive step-size control
    mesh = [t_start]
    y = np.array([y_0])       # y vector of solutions y_l
    l = 0       # counter
    while True:
        h = np.amin([t_end - mesh[l], np.amax([h_min, h])])
        F = fi_2(fun, mesh[l], y[l], h)
        nrm = np.linalg.norm(fi_1(fun, mesh[l], y[l], h) - F)
        H = rho * ((tau / nrm) ** (1 / p)) * h
        if h <= H or h <= h_min:
            mesh.append(mesh[l] + h)
            y_new = np.array([y[l] + h * F])
            y = np.append(y, y_new, axis=0)
            if mesh[-1] < t_end:
                h = np.amin([H, lamb * h])
                l = l + 1
        else:
            h = np.amin([H, h / lamb])

        if np.abs(mesh[-1] - t_end) < 1e-13:     # if t_l+1 = t_end
            break

    return mesh, y, l


# f - RHS of equation, f_y - derivative of f wrt. y, f_y_prime - derivative of f wrt. y'
# y_a, y_b - boundary conditions, a, b - interval, s_0 - s starting value

def shooting(f, f_y, f_y_prime, y_a, y_b, a, b, s_0, tol):

    def F(x, Z):        # Z = [y, y', v, v'] where v' = d/ds y
        r = np.zeros(4)
        r[0] = Z[1]
        r[1] = f(x, Z[0], Z[1])
        r[2] = Z[3]
        r[3] = f_y(x, Z[0], Z[1]) * Z[2] + f_y_prime(x, Z[0], Z[1]) * Z[3]
        return r

    s = s_0
    s_array = [s_0]
    resid_array = []
    n = 0
    mesh = []
    Z = np.array([])
    while(True):
        init = np.array([y_a, s, 0, 1])
        mesh, Z, l = ivp_solver(a, b, init, F)    # mesh - returned mesh, Z - solution of ivp, l - number of steps
        resid = Z[-1, 0] - y_b      # residuum at point b: y[b] - y_b
        update = resid/(Z[-1, 2])
        s = s - update
        s_array = s_array + [s]
        resid_array = resid_array + [np.abs(resid)]
        n = n+1


        if np.abs(update) <= tol or np.abs(resid) <= tol:
            break

    y = Z[:, 0]
    return mesh, y, s_array[0:-1], resid_array


def f(x, y, y_prime):       # RHS of boundary value problem
    l = 110
    return l*y + y_prime


def f_y(x, y, y_prime):     # d/dy f
    l = 110
    return l


def f_y_prime(x, y, y_prime):       # d/dy' f
    return 1


def s_exact(T):
    return (-10 * np.exp(21*T) + 21 * np.exp(10*T) - 11)/(np.exp(21*T) - 1)


def main():
    y_a, y_b = 1, 1
    a, b = 0, 1
    s_0 = 3
    tol = 1e-15
    c = np.linalg.solve(np.array([[np.exp(-10*a), np.exp(11*a)],[np.exp(-10*b), np.exp(11*b)]]), np.array([1, 1]))
    mesh, y, s_array, resid_array = shooting(f, f_y, f_y_prime, y_a, y_b, a, b, s_0, tol)
    plt.plot(mesh, y, label="numerical solution")
    plt.plot(mesh, list(map(lambda x: c[0] * np.exp(-10*x) + c[1] * np.exp(11*x), mesh)), label="true solution")
    plt.title("a = 0, b = 1")
    plt.legend()
    plt.show()

    m = len(s_array)
    s = s_exact(b)

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(m), list(map(lambda x: np.abs(x - s), s_array)), label="$|s_n - s_{exact}|$", color="blue")
    ax.legend(loc="upper left")
    ax.set_ylabel("s error")
    ax2 = ax.twinx()
    ax2.semilogy(np.arange(m), resid_array, label="$|y_N - 1|$", color="red")
    ax2.set_ylabel("y(b) error")
    ax2.legend(loc="upper right")
    plt.show()


    b = 10
    s_0 = 15
    tol = 1e-14
    c = np.linalg.solve(np.array([[np.exp(-10*a), np.exp(11*a)], [np.exp(-10*b), np.exp(11*b)]]), np.array([1, 1]))
    mesh, y, s_array, resid_array = shooting(f, f_y, f_y_prime, y_a, y_b, a, b, s_0, tol)
    plt.plot(mesh, y, label="numerical solution")
    plt.plot(mesh, list(map(lambda x: c[0] * np.exp(-10*x) + c[1] * np.exp(11*x), mesh)), label="true solution")
    plt.title("a = 0, b = 10")
    plt.legend()
    plt.show()

    m = len(s_array)
    s = s_exact(b)

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(m), list(map(lambda x: np.abs(x - s), s_array)), label="$|s_n - s_{exact}|$", color="blue")
    ax.legend(loc="upper left")
    ax.set_ylabel("s error")
    ax2 = ax.twinx()
    ax2.semilogy(np.arange(m), resid_array, label="$|y_N - 1|$", color="red")
    ax2.set_ylabel("y(b) error")
    ax2.legend(loc="upper right")
    plt.show()


main()