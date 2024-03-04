import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm

# matrix from task definition
def MatrixTemplate(N):
    MatrixArray = np.zeros(shape = (N, N))
    for i in range(N):
        if (i > 0):
            MatrixArray[i][i - 1] = -1
        if (i < (N - 1)):
            MatrixArray[i][i + 1] = -1
    MatrixArray[0][N - 1] = -1
    MatrixArray[N - 1][0] = -1
    for i in range(N):
        MatrixArray[i][i] = 2 + i**2 / N**2
    return MatrixArray

#free column from task definition
def RightPartTemplate(N):
    f = np.zeros(shape = (N, 1))
    for i in range(N):
        f[i, 0] = (1 + N**2 * math.sin(math.pi / N)**2) * math.sin(2 * math.pi * i / N)
    return f

# returns Gershgorin circles massive for matrix A like (center, radius)
def gershgorinCircles(A):
    Circles = []
    for i in range(len(A)):
        Center = A[i, i]
        R = 0.0
        for j in range(len(A)):
            if not j == i:
                R += math.fabs(A[i, j])
        Circles.append((Center, R))
    return Circles

# returns system matrix from Krilov method системы на Yk = A**k * Y из метода Крылова
def getMtxKrilov(A, Y):
    N = len(A)
    MtxKrilov = []
    Yk = Y
    for k in range(N, 0, -1):
        Yk = np.dot(A, Yk)
        MtxKrilov.append(np.reshape(Yk, (1, N)))
    Mtx = np.zeros(shape = (N, N ))
    for i in range(N):
        for j in range(N):
            Mtx[i, j] = MtxKrilov[i][0][j]
    return np.transpose(Mtx)

# returns nnext member in MSI for polynomial X**N * P[0] + ... + X * P[N - 1] + P[N] = 0
# MSI: X_{k+1} = (X_{k}**N * P[0] + ... + X_{k}**2 * P[N - 1] + P[N]) / (- P[N - 1])
# P - massiv with polynomial coeffitients from older
def getNextXPolynom(Xk, P):
    N = len(P) - 1
    Xk1 = P[N] / (- P[N - 1])
    for j in range(2, N + 1):
        Xk1 += P[N - j] * Xk**j / (- P[N - 1])
    return Xk1

# returns lamdas from characteristic equation A using MSI with Gershgorin circles
# P - massiv with polynomial coeffitients from older
def getLambdas(A, P, Epsilon):
    Lambdas = []
    Circles = gershgorinCircles(A)
    N = len(A)
    for i in range(len(A)):
        Xk = Circles[i][0] - Circles[i][1] / 2
        Xk1 = getNextXPolynom(Xk, P)
        while (math.fabs(Xk - Xk1) > Epsilon):
            Xk = Xk1
            Xk1 = getNextXPolynom(Xk, P)
        P1 = [1, -Xk1]
        P = np.polydiv(P, P1)[0]
        Lambdas.append(Xk1)
    return Lambdas

# returns optimum tau from lamdas massive
def getTauOpt(Lambdas):
    return 2.0 / (max(Lambdas) + min(Lambdas))

# returns optimum tau for system with matrix A
# solves eigenvalues using Krilov method
def getTauOptKrilov(A, Epsilon):
    N = len(A)
    Y = np.ones(shape = (N, 1))
    MtxKrilov = getMtxKrilov(A, Y)
    P = np.linalg.solve(MtxKrilov, Y)
    P1 = []
    for i in range(len(P) - 1, -1, -1):
        P1.append(P[i, 0])
    P1.append(-1)
    Lambdas = getLambdas(A, P1, Epsilon)
    return getTauOpt(Lambdas)

# same using numpy
def getTauOptLinalEig(A):
    Lambdas, Vectors = np.linalg.eig(A)
    return getTauOpt(Lambdas)
    
# next element in MSI
def getNextX(Xk, A, f, Tau):
    Xk1 = (np.eye(len(A)) - Tau * np.matrix(A)) * Xk + Tau * f
    return Xk1

# solves AX = f using MSI
def getMPISolution(A, f, Tau, Epsilon):
    Xk = np.ones(shape = (len(A), 1))
    Xk1 = getNextX(Xk, A, f, Tau)
    Norm = norm((Xk1 - Xk), ord=2)
    Count = 0
    Norms = []
    while (Norm > Epsilon):
        Norms.append(Norm)
        Count += 1
        Xk = Xk1
        Xk1 = getNextX(Xk, A, f, Tau)
        Norm = norm((Xk1 - Xk), ord=2)
    return Xk1, Norms

def showGraphic(Norms):
    plt.figure(figsize = (5, 5))
    plt.xlabel('Iteration number')
    plt.ylabel('Residual number')
    Iterates = np.arange(len(Norms))
    plt.plot(Iterates, Norms, 'y')
    plt.grid()
    plt.show()

def main():
    Epsilon = 1e-4
    N = 6
    A = MatrixTemplate(N)
    print("System matrix: A =\n", A)
    f = RightPartTemplate(N)
    print("Free column: f =\n", f)
    X0 = np.linalg.solve(A, f)
    print("\nExact system solution: X0 =\n", X0)
 
    TausWithNames = [(0.45, "Arbitrary Tau"),
                     (getTauOptKrilov(A, Epsilon), "Optimum Krilov"),
                     (getTauOptLinalEig(A), "Optimum np.linalg.eig")]
    for i in range(0, len(TausWithNames)):
        X, Norms = getMPISolution(A, f, TausWithNames[i][0], Epsilon)
        dX = norm(X - X0, ord=2)
        print("\n", TausWithNames[i][1], "(Tau =", TausWithNames[i][0], "):", "\nSystem solution: X =\n", X, \
              "\n\nNorm of difference ||X - X0||:", format(dX, '.10f'))
        showGraphic(Norms)
        
    
if __name__ == "__main__":
    main()
