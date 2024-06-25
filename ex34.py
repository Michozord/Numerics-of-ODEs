import numpy as np
from matplotlib import pyplot as plt


def complex_matrix(re_min, re_max, im_min, im_max, n):
    re = np.linspace(re_min, re_max, n)
    im = np.linspace(im_min, im_max, n)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j



def is_in_domain(alpha, beta, z):   # prove, if roots of polynomial rho_z (x) = rho(x) - z * sigma(x) are | | < 1 or simple if | | ==1
    tau = 1e-15
    coeffs = alpha - z * beta       # coefficients of rho_z
    while coeffs[-1] == 0:      # remove 0 coefficients
        coeffs = coeffs[0:-1]
    coeffs = (1 / coeffs[-1]) * coeffs   # normalisation of polynomial
    A_p = (1 + 0j)*np.eye(len(coeffs) - 1, k=-1)     # companion matrix
    A_p[:,-1] = -coeffs[0:-1]
    roots = np.linalg.eigvals(A_p)
    problem_roots = []      # list of roots with | | == 1
    for r in roots:
        abs_r = abs(r)
        if abs_r > 1 + tau:
            return False
        elif abs_r > 1 - tau:
            problem_roots = problem_roots + [r]
    problem_roots = sorted(problem_roots, key=lambda x : x.real)    # sort problematic roots by its real parts
    for i in range(len(problem_roots) - 1):     # proves, if problematic roots are simple
        if np.abs(problem_roots[i+1] - problem_roots[i]) < tau:
            return False
    return True


def stability_domain(re_min, re_max, im_min, im_max, alpha, beta, n, label):
    Q = complex_matrix(re_min, re_max, im_min, im_max, n)       # matrix of complex numbers (input)
    comp = lambda z: is_in_domain(alpha, beta, z)
    Q = np.array([list(map(comp, v)) for v in Q])
    plt.imshow(Q, cmap="binary", origin="lower", extent=(re_min, re_max, im_min, im_max))
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(label)
    plt.show()


n = 200
re_min, re_max, im_min, im_max = -5, 5, -5, 5

# Adams-Bashford methods:
alpha_AB0 = np.array([-1, 1])
beta_AB0 = np.array([1, 0])
alpha_AB1 = np.array([0, -1, 1])
beta_AB1 = np.array([-1/2, 3/2, 0])
alpha_AB2 = np.array([0, 0, -1, 1])
beta_AB2 = np.array([5/12, -4/3, 23/12, 0])
alpha_AB3 = np.array([0, 0, 0, -1, 1])
beta_AB3 = np.array([-3/8, 37/24, -59/24, 55/24, 0])

# Adams-Moulton methods:
alpha_AM0 = np.array([-1, 1])
beta_AM0 = np.array([0, 1])
alpha_AM1 = np.array([-1, 1])
beta_AM1 = np.array([1/2, 1/2])
alpha_AM2 = np.array([0, -1, 1])
beta_AM2 = np.array([-1/12, 2/3, 5/12])
alpha_AM3 = np.array([0, 0, -1, 1])
beta_AM3 = np.array([1/24, -5/24, 19/24, 3/8])

#BDF methods:
alpha_BDF1 = np.array([-1, 1])
beta_BDF1 = np.array([0, 1])
alpha_BDF2 = np.array([1/3, -4/3, 1])
beta_BDF2 = np.array([0, 0, 2/3])
alpha_BDF3 = np.array([-2/11, 9/11, -18/11, 1])
beta_BDF3 = np.array([0, 0, 0, 6/11])


stability_domain(re_min, re_max, im_min, im_max, alpha_AB0, beta_AB0, n, "AB0 method (Expl. Euler)")
stability_domain(re_min, re_max, im_min, im_max, alpha_AB1, beta_AB1, n, "AB1 method")
stability_domain(re_min, re_max, im_min, im_max, alpha_AB2, beta_AB2, n, "AB2 method")
stability_domain(re_min, re_max, im_min, im_max, alpha_AB3, beta_AB3, n, "AB3 method")

stability_domain(re_min, re_max, im_min, im_max, alpha_AM0, beta_AM0, n, "AM0 method (impl. Euler)")
stability_domain(re_min, re_max, im_min, im_max, alpha_AM1, beta_AM1, n, "AM1 method (trapezoidal)")
stability_domain(re_min, re_max, im_min, im_max, alpha_AM2, beta_AM2, n, "AM2 method")
stability_domain(re_min, re_max, im_min, im_max, alpha_AM3, beta_AM3, n, "AM3 method")

stability_domain(re_min, re_max, im_min, im_max, alpha_BDF1, beta_BDF1, n, "BDF1 method")
stability_domain(re_min, re_max, im_min, im_max, alpha_BDF2, beta_BDF2, n, "BDF2 method")
stability_domain(re_min, re_max, im_min, im_max, alpha_BDF3, beta_BDF3, n, "BDF3 method")

