import numpy as np
import math
import matplotlib.pyplot as plt

def g(X : float) -> float:
    return 1

def f(X : float) -> float:
    return math.cos(X * math.pi)

def K(X : float, S : float) -> float:
    return 0.2 / (0.04 + (X - S)**2)

# Finds the Lagrange basis function lk(t)
# ArgValues ​​- array of n argument values ​​[t0, ... tn]
def getl(t : float, k : int, ArgValues : list) -> float:
    n = len(ArgValues)
    lk = 1
    for j in range(n):
        Denom = ArgValues[k] - ArgValues[j]
        lk *= ((t - ArgValues[j]) / Denom) if k != j else 1
    return lk

# Returns an array of N equally spaced points in the segment [Start, Stop]
def divideSegment(Start : float, Stop : float, N : int) -> float:
    Points = []
    Step = (Stop - Start) / (N - 1)
    for PointNum in range(N):
        Points.append(Start + PointNum * Step)
    return Points

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

# Returns the coefficients of the Newton interpolation polynomial (divided differences)
def getNewtonPolinom(ArgValues : list, FunctionValues : list) -> list:
    n = len(ArgValues)
    DivDiffs = []
    for k in range(n):
        F = getF(0, k, ArgValues, FunctionValues)
        DivDiffs.append(F)
    return DivDiffs

# Returns the value of Newton's polynomial at point t
def getNewtonValue(t : float, ArgValues : list, NewtonPol : list) -> float:
    Result = 0
    n = len(ArgValues)
    for k in range(n):
        Mult = 1
        for i in range(k):
            Mult *= (t - ArgValues[i])
        Result += Mult * NewtonPol[k]
    return Result

# Returns an array of Newton polynomial values ​​for each Args value
# ArgValues ​​- array of n values ​​of argument t -- [t0, ... tn]
# FunctionValues ​​- an array of n function values ​​at points [t0, ... tn]
def getNewtonValues(Args : list, ArgValues : list, FunctionValues : list) -> list:
    NewtonValues = []
    NewtonPol = getNewtonPolinom(ArgValues, FunctionValues)
    for Arg in Args:
        NewtonValues.append(getNewtonValue(Arg, ArgValues, NewtonPol))
    return NewtonValues

# Finds the divided difference F(tk, ... tn)
# ArgValues ​​- array of n argument values ​​[t0, ... tn]
# FunctionValues ​​- array of n - k + 1 function values ​​at points [tk, ... tn]
def getF(k : int, n : int, ArgValues : list, FunctionValues : list) -> float:
    if (k == n):
        return FunctionValues[0]
    F2 = getF(k + 1, n, ArgValues, FunctionValues[1:])
    F1 = getF(k, n - 1, ArgValues, FunctionValues[:-1])
    t2 = ArgValues[n]
    t1 = ArgValues[k]
    DivDiff = (F2 - F1) / (t2 - t1)
    return DivDiff

# Calculates the integral of a pointwise given function with values ​​Functions[i] on the interval [a, b] using the trapezoidal method
def calculateTrapezoidByPoints(Functions : list, a : float, b : float) -> float:
    Int = 0
    N = len(Functions)
    h = (b - a) / N
    for i in range(1, N):
        F1 = Functions[i]
        F2 = Functions[i - 1]
        Int += h / 2.0 * (F1 + F2)
    return Int

# Matrix decomposition into product A = LU
# U - upper triangular, L - lower triangular
def LUdecomposition(A : list) -> (list, list):
    N = len(A)
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

# Solve SLAE UX = F
# U - upper triangular matrix
def solveU(U : list, F : list) -> list:
    N = len(U)
    Solution = np.zeros(N)
    Solution[N - 1] = F[N - 1, 0] * 1.0 / U[N - 1][N - 1]
    for n in range(N - 2, -1, -1):
        Sum = 0
        for i in range(n + 1, N):
            Sum += U[n][i] * Solution[i]
        Solution[n] = (F[n, 0] - Sum) / U[n][n]
    return Solution

# Solve SLAE
def getSolution(Matrix : list, F : list) -> list:
    N = len(Matrix)
    L, U = LUdecomposition(Matrix)
    F = np.dot(np.linalg.inv(L), F)
    return solveU(U, F)

# Returns
# Nodes - an array of N + k points xk, obtained by equal division of the segment [a, b] into N - 1 parts and supplemented with k points
# FunctionValues ​​- array of function values ​​u(x) at these points
def getFunctionValues(Kfunc : "function", g : "function", f : "function", Lambda : float, a : float, b : float, N : int, ArrayX0 : list) -> (list, list):
    
    # 1. Divide the segment into N-1 parts
    Nodes = divideSegment(a, b, N)
    
    # 2. We supplement it so that the maximum of X0 falls inside the interpolation interval
    h = (b - a) / (N - 1)
    K = int((max(ArrayX0) - b) / h) + 1
    AddNodes = [b + i * h for i in range(1, K + 1)]
    Nodes += AddNodes
    
    # 3. Looking for the Weights of the quadrature formula by integrating the basic Lagrange polynomials
    Weights = []
    for k in range(1, N + 1):
        # Integration by trapezoidal method over 1000 points
        Args = np.arange(a, b, 0.001)
        BaseLagranValues = [getl(i, k - 1, Nodes) for i in Args]
        Weights.append(calculateSimpsonByPoints(BaseLagranValues, a, b))
        
    # 4. We obtain the matrix of the SLAE Matrix and the column of the right sides F
    Matrix = np.zeros(shape = (N + K, N + K))
    for i in range(N + K):
        for j in range(N + K):
            if (j < N):
                Matrix[i, j] = -Lambda * Weights[j] * Kfunc(Nodes[i], Nodes[j])
    for i in range(N + K):
        Matrix[i][i] += g(Nodes[i])
        
    F = np.zeros(shape = (N + K, 1))
    for i in range(N + K):
        F[i, 0] = f(Nodes[i])

    # 5. Solving SLAE
    FunctionValues = getSolution(Matrix, F)
    
    return Nodes, FunctionValues

def getFunctionValue(X0 : float, ArgValues : list, FunctionValues : list) -> float:
    NumArg = 0
    while (ArgValues[NumArg] < X0):
        NumArg += 1
    return FunctionValues[NumArg]
    
        
def main():
    a = -1
    b = 1
    Lambda = -1
    # Number of nodes
    ArrayN = [3, 4, 5, 6]
    # Points at which we want to get the value u(x)
    ArrayX0 = [1.1, 1.25, 1.5]
    
    plt.figure(figsize = (7.5, 7.5))
    plt.title("Newton's interpolation polynomial for the function u(x)")
    
    for N in ArrayN:
        ArgValues, FunctionValues = getFunctionValues(K, g, f, Lambda, a, b, N, ArrayX0)
        plt.scatter(ArgValues, FunctionValues, marker = "^")
        Args = np.arange(-1, max(ArgValues) + 0.01, 0.01)
        NewtonVals = getNewtonValues(Args, ArgValues, FunctionValues)
        print("\n\nValues at points by constructing a polynomial over n =", N, "nodes:\n")
        for X0 in ArrayX0:
            print("X0 =", X0, ", u(X0) =", format(getFunctionValue(X0, Args, NewtonVals), '.4f'), "\n")

        NameGraph = "Newton n = " + str(N)
        plt.plot(Args, NewtonVals, label = NameGraph)

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
