import numpy as np
import math
import matplotlib.pyplot as plt

#returns system matrix that fits the size
def Matrix(N):
    MatrixArray = np.zeros(shape = (N, N))
    for i in range(N - 1):
        MatrixArray[i][i] = 1
        for j in range(i + 1, N):
            MatrixArray[i][j] = -1
    for j in range(N):
        MatrixArray[N - 1][j] = 1
    return MatrixArray

#returns column with free members
def RightPart(N):
    return np.ones(shape = (N, 1))

#Third norm
def ThirdNorm(Mtx, TMtx):
    Mult = np.dot(TMtx, Mtx)
    MaxL = 0
    Lambdas, Vectors = np.linalg.eig(Mult)
    for l in Lambdas:
        if (l > MaxL):
            MaxL = l
    return math.sqrt(MaxL)

#returns number of conditions
def Mu3(N):
    Mtx = Matrix(N)
    RMtx = np.linalg.inv(Mtx)
    return ThirdNorm(Mtx, np.transpose(Mtx)) * ThirdNorm(RMtx, np.transpose(RMtx))

# decomposes A to LU
# U - uppertriangular, L - inferiortriangular
def LUdecomposition(N, A):
    L = np.zeros(shape = (N, N))
    U = A
    for i in range(N):
        for j in range(i, N):
            L[j][i] = U[j][i] * 1.0 / U[i][i]

    for k in range(1, N):
        for i in range(k-1, N):
            for j in range(i, N):
                L[j][i] = U[j][i] * 1.0 / U[i][i]
        for i in range(k, N):
            for j in range(k-1, N):
                U[i][j] = U[i][j] - L[i][k-1] * U[k-1][j]
    return (L, U)

# solves UX = f
def solveU(N, U, f):
    Solution = np.zeros(N)
    Solution[N - 1] = f[N - 1, 0] * 1.0 / U[N - 1][N - 1]
    for n in range(N - 2, -1, -1):
        Sum = 0
        for i in range(n + 1, N):
            Sum += U[n][i] * Solution[i]
        Solution[n] = (f[n, 0] - Sum) / U[n][n]
    return Solution

#returns solution for given N
def getSolution(N):
    f = RightPart(N)
    A = Matrix(N)
    L, U = LUdecomposition(N, A)
    f = np.dot(np.linalg.inv(L), f)
    return solveU(N, U, f)

def main():
    plt.figure(100)
    plt.title("Number of conditions from n in Euclidean norm")

    plt.xlabel('n')
    plt.ylabel('Mu')

    N = 15
    X = np.arange(N)
    Y = [Mu3(i) for i in X]

    A, B = np.polyfit(X, Y, 1)
    YL = [A * x + B for x in X]

    for i in range(1, N + 1):
        print("Matrix for n = ", i, " : \n", Matrix(i), "\nSLE solution X =", getSolution(i), "\n")

    plt.scatter(X, Y)
    plt.plot(X, YL, 'b')

    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
