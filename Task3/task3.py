import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm

#returns system matrix
def Matrix(N, a):
    MatrixArray = np.zeros(shape = (N, N))
    for i in range(N):
        MatrixArray[i][i] = 2
        if (i != (N - 1)):
            MatrixArray[i][i + 1] = -1 - a
        if (i != 0):
            MatrixArray[i][i - 1] = -1 + a
    return MatrixArray

#returns column with free members
def RightPart(N, a):
    f = np.zeros(shape = (N, 1))
    f[0][0] = 1 - a
    f[N - 1][0] = 1 + a
    return f

# decomposes Mtx to L + D + U
#  L - inferiortriangular, U - uppertriangular, D - diagonal
def Decomposition(N, Mtx):
    L = np.zeros(shape = (N, N))
    D = np.zeros(shape = (N, N))
    U = np.zeros(shape = (N, N))
    for i in range(N):
        for j in range(i):
            L[i][j] = Mtx[i][j]
    for i in range(N):
        for j in range(i):
            U[j][i] = Mtx[j][i]
    for i in range(N):
        D[i][i] = Mtx[i][i]
    return (L, D, U)

#returns next element
def nextX(LDinv, f, U, Xk):
    Xk1 = np.matrix(LDinv) * np.matrix(f - np.dot(U, Xk))
    return Xk1

# solves SLE using Zeidel method with Epsilon accuracy
# returnns x and number iterations
def solveZeidel(N, Mtx, f, Epsilon):
    L, D, U = Decomposition(N, Mtx)
    Cnt = 0
    LDinv = np.linalg.inv(L + D)
    Xk = np.zeros(shape = (N, 1))
    Xk1 = nextX(LDinv, f, U, Xk)
    while (norm((Xk1 - Xk), ord=2) > Epsilon):
        Cnt += 1
        Xk = Xk1
        Xk1 = nextX(LDinv, f, U, Xk)
    return (np.reshape(Xk1, (1, N)), Cnt)

def main():
    N = 15
    a = 0.1
    eps = 1e-5
    Mtx = Matrix(N, a)
    f = RightPart(N, a)
    X, Cnt = solveZeidel(N, Mtx, f, eps)

    print("Iteration till ||x(k) - x(k+1)|| > ", eps)
    print("SLE solution for n =", N, ", a = ", a, ": X = ", X, "\n")

    plt.figure(100)
    plt.title("Number of iterations from a-parameter for n = 15")

    plt.xlabel('a')
    plt.ylabel('Cnt')

    Alphas = np.arange(-1.4, 1.5, 0.1)
    Cnt = len(Alphas)
    Y = np.zeros(Cnt)
    for i in range(Cnt):
        Mtx = Matrix(N, Alphas[i])
        f = RightPart(N, Alphas[i])
        X, Cnt = solveZeidel(N, Mtx, f, eps)
        Y[i] = Cnt

    plt.scatter(Alphas, Y)
    plt.plot(Alphas, Y, 'b')

    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
