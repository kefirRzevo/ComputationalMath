import numpy as np
import math
import os
import matplotlib.pyplot as plt

# Function from task 1
def Function1(X : float) -> float:
    return 1.0 / (1 + 25 * X * X)

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

# Returns an array of N equally spaced points in the segment [Start, Stop]
def divideSegment(Start : float, Stop : float, N : int) -> float:
    Points = []
    Step = (Stop - Start) / (N - 1)
    for PointNum in range(N):
        Points.append(Start + PointNum * Step)
    return Points

def main1():
    StartSegment = -1
    EndSegment = 1

    plt.figure(figsize = (7.5, 7.5))
    plt.title("Lagrange interpolation polynomial for different n")

    Args = np.arange(-1, 1.01, 0.01)
    Vals = [Function1(Arg) for Arg in Args]
    plt.plot(Args, Vals, 'r', label = "Plot of original function")
    
    ArrayN = [4, 6, 10]
    
    for N in ArrayN:
        ArgValues = divideSegment(StartSegment, EndSegment, N)
        FunctionValues = []
        for Arg in ArgValues:
            FunctionValues.append(Function1(Arg))
        plt.scatter(ArgValues, FunctionValues, marker = "^")
        Args = np.arange(-1, 1.01, 0.01)
        LagrangeVals = getLagrangeValues(Args, ArgValues, FunctionValues)
        NameGraph = "Lagrange n = " + str(N)
        plt.plot(Args, LagrangeVals, label = NameGraph)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Lagrange.png'))
    plt.show()

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

# Returns an array of N zeros of the Chebyshev polynomial on the segment [Start, Stop]
def getChebZeros(Start : float, Stop : float, N : int) -> float:
    Zeros = []
    HalfSum = (Start + Stop) / 2.0
    HalfDiff = (Stop - Start) / 2.0
    for ZeroNum in range(1, N + 1):
        Zero = HalfSum + HalfDiff * math.cos((2 * ZeroNum - 1) * math.pi / (2 * N))
        Zeros.append(Zero)
    return Zeros

def main2():
    StartSegment = -1
    EndSegment = 1
    
    plt.figure(figsize = (7.5, 7.5))
    plt.title("Newton's interpolation polynomial with nodes at the zeros of the Chebyshev polynomial for different n")
    
    Args = np.arange(-1, 1.01, 0.01)
    Vals = [Function1(Arg) for Arg in Args]
    plt.plot(Args, Vals, 'r', label = "Plot of original function")
    
    ArrayN = [4, 6, 10]
    
    for N in ArrayN:
        NewtonArgValues = getChebZeros(StartSegment, EndSegment, N)
        FunctionValues = []
        for Arg in NewtonArgValues:
            FunctionValues.append(Function1(Arg))
        plt.scatter(NewtonArgValues, FunctionValues, marker = "^")
        Args = np.arange(-1, 1.01, 0.01)
        NewtonVals = getNewtonValues(Args, NewtonArgValues, FunctionValues)
        NameGraph = "Newton n = " + str(N)
        plt.plot(Args, NewtonVals, label = NameGraph)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'NewtonWithCheb.png'))
    plt.show()


if __name__ == '__main__':
    main1()
    main2()
