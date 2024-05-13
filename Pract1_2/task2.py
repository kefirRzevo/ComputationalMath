import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

# Finds the Lagrange basis function lk(t)
# ArgValues ​​- array of n argument values ​​[t0, ... tn]
def getl(t : float, k : int, ArgValues : list) -> float:
    n = len(ArgValues)
    lk = 1
    for j in range(n):
        Denom = ArgValues[k] - ArgValues[j]
        lk *= ((t - ArgValues[j]) / Denom) if k != j else 1
    return lk

# Returns the value of the Lagrange polynomial at point t
def getLagrangePolinom(t : float, ArgValues : list, FunctionValues : list) -> float:
    n = len(ArgValues)
    Value = 0
    for k in range(n):
        Value += (getl(t, k, ArgValues) * FunctionValues[k])
    return Value

# Returns an array of Lagrange polynomial values ​​for each Args value
# ArgValues ​​- array of n values ​​of argument t -- [t0, ... tn]
# FunctionValues ​​- an array of n function values ​​at points [t0, ... tn]
def getLagrangeValues(Args : list, ArgValues : list, FunctionValues : list) -> list:
    Values = []
    for Arg in Args:
        Values.append(getLagrangePolinom(Arg, ArgValues, FunctionValues))
    return Values

# Function from task 2
def Function2(X : float) -> float:
    return math.log(100.0 - X) / (10.0 - math.sqrt(X))

# Calculate the integral of a pointwise given function with values ​​Functions[i] on the interval [a, b] using the Simpson method
def calculateSimpsonByPoints(Functions : list, a : float, b : float) -> float:
    Int = 0
    N = len(Functions)
    k = int(N / 2)
    h = (b - a) / N
    for i in range(1, k):
        F1 = Functions[2*i]
        F2 = Functions[2*i - 1]
        F3 = Functions[2*i - 2]
        Int += h / 3.0 * (F1 + 4 * F2 + F3)
    return Int

# Returns the next term in Newton's iterative method
# The process is represented by the following iterative relation: X_{k+1} = X_{k} - P(Xk) / P1(Xk)
# P - the value of the Legendre polynomial for Xk
# P1 - the value of the derivative of the Legendre polynomial for Xk
def getNextNewtonIteration(Xk : float, P : float, P1 : float) -> float:
    return Xk - P / P1

# Returns the Nth Legendre polynomial for X
def getLejanPol(X : float, N : int) -> float:
    if (N == 0):
        return 1
    if (N == 1):
        return X
    return (2.0 * N + 1) * X * getLejanPol(X, N - 1) / (N + 1) - N * getLejanPol(X, N - 2) / (N + 1)

# Returns the first derivative of the Nth Legendre polynomial for X
def getLejanDerr(X : float, N : int) -> float:
    return N * (getLejanPol(X, N - 1) - X * getLejanPol(X, N)) / (1 - X * X)

# Returns the N zeros of the Legendre polynomial
#, calculated iteratively using Newton’s method with the initial approximation: X0 = cos(pi(4i - 1)/(4N + 2))
def getLejanZeros(N : int) -> list:
    Zeros = []
    Epsilon = 1e-3
    for i in range(1, N + 1):
        Xk = math.cos(math.pi * (4 * i - 1) / (4 * N + 2))
        Xk1 = getNextNewtonIteration(Xk, getLejanPol(Xk, N), getLejanDerr(Xk, N))
        while (abs(Xk - Xk1) > Epsilon):
            Xk = Xk1
            Xk1 = getNextNewtonIteration(Xk, getLejanPol(Xk, N), getLejanDerr(Xk, N))
        Zeros.append(Xk1)
    return Zeros

# Makes a change of variables at the quadrature nodes to move from integration
# over [-1, 1] to integration over [a, b]
def changeVars(Start : float, Stop : float, Vars : list) -> list:
    HalfSum = (Start + Stop) / 2.0
    HalfDiff = (Stop - Start) / 2.0
    NewVars = [HalfSum + HalfDiff * T for T in Vars]
    return NewVars

# Calculate the integral of F on the interval [a, b] using the Gaussian quadrature method over N nodes
def calculateWithGaussQuadrature(F : "function", a : float, b : float, N : int) -> float:
    Int = 0
    NodesT = getLejanZeros(N)
    # Integration by Simpson's method over 1000 points
    NodesX = changeVars(a + 1e-1, b, NodesT)
    for k in range(1, N + 1):
        Args = np.arange(a, b, 0.001)
        BaseLagranValues = [getl(i, k - 1, NodesX) for i in Args]
        Ck = calculateSimpsonByPoints(BaseLagranValues, a, b)
        Fk = F(NodesX[k - 1])
        Int += Ck * Fk
    return Int

def main():
    StartSegment = 0
    EndSegment = 10
    
    ArrayN = np.arange(2, 20)
    Errors =[]
    
    Exact = scipy.integrate.quad(Function2, StartSegment, EndSegment)
    print("Exact solution and its error (I, delta I) =", Exact)
    
    for N in ArrayN:
        Gauss = calculateWithGaussQuadrature(Function2, StartSegment, EndSegment, N)
        print("\nGaussian quadrature method with N =", N, ": I =", format(Gauss, '.10f'))
        Errors.append(abs(Gauss - Exact[0]) * 100 / Exact[0])

    plt.figure(figsize = (7.5, 7.5))
    
    plt.title("Curve of the dependence of the relative integration error on the number of nodes,\n interpolated by the Lagrange polynomial:)")
    plt.scatter(ArrayN, Errors, marker = '^', color = 'r')
    Args = np.arange(min(ArrayN), max(ArrayN), 0.01)
    NewtonVals = getLagrangeValues(Args, ArrayN, Errors)
    plt.plot(Args, NewtonVals, color = 'g')
    plt.xlabel('Number of nodes n')
    plt.ylabel('Error, %')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'res/Gauss.png'))
    plt.show()

if __name__ == '__main__':
    main()
