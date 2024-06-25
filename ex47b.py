import numpy as np
from matplotlib import pyplot as plt

def Cholesky(h, N_1, N_2):
    N = N_1 * N_2
    A = 4 * np.eye(N) - np.eye(N, k=-1) - np.eye(N, k=1) - np.eye(N, k=-N_1) - np.eye(N, k=N_1)
    A = 1/(h*h) * A
    #print(A)
    plt.imshow(A)
    plt.title("A matrix")
    plt.show()

    L = np.linalg.cholesky(A)
    #print(L)
    plt.imshow(L)
    plt.title("L matrix")
    plt.show()



Cholesky(1, 5, 3)
