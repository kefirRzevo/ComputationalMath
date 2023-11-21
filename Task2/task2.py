import numpy as np
import math
import matplotlib.pyplot as plt

def Matrix(N):
    MatrixArray = np.zeros(shape = (N, N))
    for i in range(N):
        MatrixArray[i][i] = 2
        if (i != 0):
            MatrixArray[i][i - 1] = -1
        if (i != (N - 1)):
            MatrixArray[i][i + 1] = -1
    return np.matrix(MatrixArray)

def Norm1(matrix):
    N = len(matrix)
    SumOfRows = np.zeros(N)
    for i in range(N):
        for j in range(N):
            SumOfRows[i] += abs(matrix[i, j])
    MaxSum = 0
    for Sum in SumOfRows:
        if (Sum > MaxSum):
            MaxSum = Sum
    return MaxSum

def Norm2(matrix):
    N = len(matrix)
    SumOfColumns = np.zeros(N)
    for i in range(N):
        for j in range(N):
            SumOfColumns[i] += abs(matrix[j, i])
    MaxSum = 0
    for Sum in SumOfColumns:
        if (Sum > MaxSum):
            MaxSum = Sum
    return MaxSum

def Norm3(matrix, Tmatrix):
    Mult = np.dot(Tmatrix, matrix)
    MaxL = 0
    Lambdas, Vectors = np.linalg.eig(Mult)
    for l in Lambdas:
        if (l > MaxL):
            MaxL = l
    return math.sqrt(MaxL)

def Mu1(N):
    matrix = Matrix(N)
    return Norm1(matrix) * Norm1(np.linalg.inv(matrix))

def Mu2(N):
    matrix = Matrix(N)
    return Norm2(matrix) * Norm2(np.linalg.inv(matrix))

def Mu3(N):
    matrix = Matrix(N)
    Rmatrix = np.linalg.inv(matrix)
    return Norm3(matrix, np.transpose(matrix)) * Norm3(Rmatrix, np.transpose(Rmatrix))

plt.figure(100)
plt.title("mu(n): green - norms 1 and 2, blue - norm 3")

plt.xlabel('n')
plt.ylabel('Mu')

X = np.arange(25)

Y1 = [Mu1(i) for i in X]
plt.scatter(X, Y1)
plt.plot(X, Y1, 'g')

Y1 = [Mu2(i) for i in X]
plt.scatter(X, Y1)
plt.plot(X, Y1, 'g')

Y1 = [Mu3(i) for i in X]
plt.scatter(X, Y1)
plt.plot(X, Y1, 'b')

plt.grid()
plt.show()
