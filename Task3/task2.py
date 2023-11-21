import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm

#returns system matrix
def Matrix():
    MatrixArray = np.zeros(shape = (2, 2))
    MatrixArray[0][0] = 0.780
    MatrixArray[0][1] = 0.563
    MatrixArray[1][0] = 0.457
    MatrixArray[1][1] = 0.330
    return MatrixArray

#Euclidian norm
def ThirdNorm(Mtx, TMtx):
    Mult = np.dot(TMtx, Mtx)
    MaxL = 0
    Lambdas, Vectors = np.linalg.eig(Mult)
    for l in Lambdas:
        if (l > MaxL):
            MaxL = l
    return math.sqrt(MaxL)

#returns number of conditions
def Mu3():
    Mtx = Matrix()
    RMtx = np.linalg.inv(Mtx)
    return ThirdNorm(Mtx, np.transpose(Mtx)) * ThirdNorm(RMtx, np.transpose(RMtx))

#returns next element
def nextX(A, Xk, f):
    Xk1 = (np.eye(2) - np.matrix(A)) * Xk + f
    return Xk1

#takes df and returns x solution of equation 
#Ax = f = f0 + dfN = RightPart(N) (Epsilon error)
def getSolution(f, Epsilon):
    A = Matrix()
    Xk = np.matrix('1; 1')
    Xk1 = nextX(A, Xk, f)
    while (norm((Xk1 - Xk), ord=2) > Epsilon):
        Xk = Xk1
        Xk1 = nextX(A, Xk, f)
    return Xk1

def main():
    eps = 1e-5
    X0 = np.matrix('1; -1')
    NormX0 = norm(X0, ord=2)
    Mu = Mu3()

    print("Iteration till ||x(k) - x(k+1)|| > ", eps)

    f0 = np.matrix('0.217; 0.127')
    NormF0 = norm(f0, ord=2)
    df1 = np.matrix('0; 0.0005')
    df2 = np.matrix('0.0001; 0')
    df3 = np.matrix('0.001; 0.0006')
    Fvectors = [f0, f0 + df1, f0 + df2, f0 + df3]

    for i in range(1, 4):
        X = getSolution(Fvectors[i], eps)
        dX = norm(X - X0, ord=2)
        print("Case", i, ":\nSLE solution: X = (", format(X[0, 0], '.3f'), ",", format(X[1, 0], '.3f'), ")", "\n||X - X0|| = ", format(dX,  '.3f'))
        print("||dX||/||X|| <= Mu(A)||df||/||f||, where Mu(A) =", format(Mu, '.0f'))
        print(format(dX / NormX0, '.2f'), " <= ", format(Mu * norm(Fvectors[i] - f0, ord=2) / NormF0, '.2f'), "\n")

if __name__ == "__main__":
    main()
