import numpy as np
from matplotlib import pyplot as plt


# t_start, t_end - time interval, y_0 - initial value,
# fun - right side of ode
# fi_1, f_2 - RK methods of order p and p+1, tau - tolerance (>0), p - order of fi_1
# h - initial step size (>0), h_min - minimal step size (>0),
# lamb - conformity factor (>=1), rho - safety factor (0 < rho <=1)
def assc(t_start, t_end, y_0, fun, fi_1, fi_2, p, h, tau=1e-1, h_min=1e-5, lamb=2, rho=0.8):     # adaptive step-size control
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
        #if l > 1:
            #print(mesh[-1] - mesh[-2], mesh[-1])

    return mesh, y, l


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


def f(t, y):
    alpha, beta, gamma, delta = 2, 0.001, 0.001, 1
    preys = alpha * y[0] - beta * y[0] * y[1]       # y_1 function - amount of preys
    predators = gamma * y[0] * y[1] - delta * y[1]  # y_2 function - amount of predators
    return np.array([preys, predators])     # [preys, predators]


def solve():
    y_0 = np.array([300, 150])
    mesh, y, l = assc(0, 20, y_0, f, RK4, RK5, 4, 1e-2)
    # mesh, y, l = assc(0, 20, y_0, f, RK4, RK5, 4, 1, tau=0.1, h_min=0.1)
    h = [mesh[j + 1] - mesh[j] for j in range(l)]

    plt.plot(mesh, y[:, 0], label="preys", color="green")
    plt.plot(mesh, y[:, 1], label="predators", color="red")
    plt.legend()
    plt.show()

    plt.plot(mesh[0:-2], h, label="step-size", color="black")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(mesh, y[:, 0], label="preys", color="green")
    ax.plot(mesh, y[:, 1], label="predators", color="red")
    ax.set_ylabel("population")
    ax.set_xlabel("time")
    ax2 = ax.twinx()
    ax2.plot(mesh[0:-2], h, label="step-size", color="black")
    ax2.set_ylabel("step-size")
    plt.show()


solve()
