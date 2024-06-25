import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve


def complex_matrix(re_min, re_max, im_min, im_max, n):
    re = np.linspace(re_min, re_max, n)
    im = np.linspace(im_min, im_max, n)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def stability_function(A, b):
    n = A.shape[0]

    def inner(z):       # R(z) = 1 + z * b^T @ (Id - z*A)^-1 @ 1
        X = np.eye(n) - z * A
        lu, piv = lu_factor(X)
        y = lu_solve((lu, piv), np.ones(n))     # y = (Id - z*A)^-1 * 1
        r = z * b.transpose() @ y          # r = z * b^T @ y
        return r + 1

    return inner


def stability_domain(re_min, re_max, im_min, im_max, A, b, n, label):
    Q = complex_matrix(re_min, re_max, im_min, im_max, n)
    R = stability_function(A, b)
    comp = lambda z: np.abs(R(z)) <= 1
    Q = np.array([list(map(comp, v)) for v in Q])
    plt.imshow(Q, cmap="binary", origin="lower", extent=(re_min, re_max, im_min, im_max))
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(label)
    plt.show()


n = 200
re_min, re_max, im_min, im_max = -5, 5, -5, 5
A_Heun = np.array([[0, 0], [1, 0]])
b_Heun = np.array([0.5, 0.5])
A_exEuler = np.array([[0]])
b_exEuler = np.array([1])
A_imEuler = np.array([[1]])
b_imEuler = np.array([1])
A_immid = np.array([[0.5]])
b_immid = np.array([1])
A_Rad = np.array([[5/12, -1/12], [3/4, 1/4]])
b_Rad = np.array([0.75, 0.25])

stability_domain(re_min, re_max, im_min, im_max, A_Heun, b_Heun, n, "Heun method")
stability_domain(re_min, re_max, im_min, im_max, A_exEuler, b_exEuler, n, "explicit Euler method")
stability_domain(re_min, re_max, im_min, im_max, A_imEuler, b_imEuler, n, "implicit Euler method")
stability_domain(re_min, re_max, im_min, im_max, A_immid, b_immid, n, "implicit midpoint method")
stability_domain(re_min, re_max, im_min, im_max, A_Rad, b_Rad, n, "2-stage Radau (implicit) method")

