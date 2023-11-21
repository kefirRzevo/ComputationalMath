import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm

# matrix from task definition
def MatrixTemplate(N):
    MatrixArray = np.zeros(shape = (N, N))
    for i in range(N):
        for j in range(N):
            MatrixArray[i, j] = 10 / (i + j + 1)
    return MatrixArray

#free column from task definition
def RightPartTemplate(N):
    Matrix = MatrixTemplate(N)
    f = np.zeros(shape = (N, 1))
    for i in range(N):
        for j in range(N):
            f[j, 0] += Matrix[i, j]
    return f

#system matrix
def Matrix():
    MatrixArray = np.zeros(shape = (2, 2))
    MatrixArray[0][0] = 0.780
    MatrixArray[0][1] = 0.457
    MatrixArray[1][0] = 0.457
    MatrixArray[1][1] = 0.330
    return MatrixArray

#free column from task definition
def RightPart(N):
    return np.ones(shape = (N, 1))

# decomposes A to L * L^T
# L - lowertriangular
def LLDecomposition(A):
    N = len(A)
    L = np.zeros(shape = (N, N))
    for i in range(N):
        for k in range(i+1):
            Sum = sum(L[i][j] * L[k][j] for j in range(k))
            if (i == k):
                L[i][k] = math.sqrt(A[i][i] - Sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - Sum))
    print("System matrix: \nA =\n", A, "\n\nLowertriangular matrix: \nL =\n", L, \
          "\nComposition: \nL*L^T =\n", np.matrix(L) * np.transpose(np.matrix(L)))
    return L

# solves UX = f
# U - uppertriangular
def solveU(U, f):
    N = len(U)
    Solution = np.zeros(N)
    Solution[N - 1] = f[N - 1, 0] * 1.0 / U[N - 1][N - 1]
    for n in range(N - 2, -1, -1):
        Sum = 0
        for i in range(n + 1, N):
            Sum += U[n][i] * Solution[i]
        Solution[n] = (f[n, 0] - Sum) / U[n][n]
    return Solution

# solves UX = f using Holec method
# A - symmetrical
def HolecSolution(A, f):
    L = LLDecomposition(A)
    f = np.dot(np.linalg.inv(L), f)
    return solveU(np.transpose(L), f)

def main():
    N = 6
    A = MatrixTemplate(N)
    f = RightPartTemplate(N)
    X = HolecSolution(A, f)
    print("\n\nMy solution X =", np.reshape(X, (len(A), 1)))
    Solution = np.linalg.solve(A, f)
    print("\n\nExact solution X0 =", Solution)
    print("\n\nNorm of difference ||X - X0||:", norm((np.reshape(X, (len(A), 1)) - Solution), 2))

if __name__ == "__main__":
    main()
