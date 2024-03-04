import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp
from scipy. integrate import odeint

def getXn(x1, x2, N):
    X = []
    h = (x2 - x1) / N
    for i in range(1, N, 1):
        X.append(x1 + i * h)
    return X

def getLambdas(x1, x2, N):
    L = []
    X = x2 - x1
    h = X * 1.0 / N
    for i in range(1, N, 1):
        L.append(4.0 / (h * h) * math.sin(math.pi * i * h / 2.0 / X) ** 2)
    return L

def getOwnFunctions(x1, x2, N, Xn):
    F = []
    X = x2 - x1
    for i in range(1, N, 1):
        F.append(math.sqrt(2.0 / X) * math.sin(math.pi * i * Xn / X))
    return F

def getG(X):
    G = []
    for x in X:
        G.append(x ** 3)
    return G
            
def getSolution(x1, x2, X, Ck_, Lambdas):
    N = len(X) + 1
    U = []
    for i in range(1, N, 1):
        OwnF = getOwnFunctions(x1, x2, N, X[i - 1])
        U.append(0)
        for j in range(1, N, 1):
            U[i - 1] += Ck_[j - 1] / Lambdas[j - 1] * OwnF[j - 1]
    return U

def main():
    x0 = 0
    x1 = 1
    N = 100
    X = getXn(x0, x1, N)
    Lambdas = getLambdas(x0, x1, N)

    Matrix = []
    for i in range(1, N, 1):
        Matrix.append(getOwnFunctions(x0, x1, N, X[i - 1]))

    G = getG(X)
    Ck_ = np.linalg.solve(Matrix, G)
    u = getSolution(x0, x1, X, Ck_, Lambdas)

    plt.figure(figsize = (8, 6))
    plt.title("Solving the problem using the Fourier method:")

    plt.plot(X, u, 'c', label = "u(x)")

    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    main()
