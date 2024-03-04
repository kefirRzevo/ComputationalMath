import numpy as np
import math
import matplotlib.pyplot as plt

# Finds the divided difference F(tk, ... tn)
# ArgValues ​​- array of n argument values ​​[t0, ... tn]
# FunctionValues ​​- array of n - k + 1 function values ​​at points [tk, ... tn]
def getF(k, n, ArgValues, FunctionValues):
    if (k == n):
        return FunctionValues[0]
    F2 = getF(k + 1, n, ArgValues, FunctionValues[1:])
    F1 = getF(k, n - 1, ArgValues, FunctionValues[:-1])
    t2 = ArgValues[n]
    t1 = ArgValues[k]
    DivDiff = (F2 - F1) / (t2 - t1)
    return DivDiff

# Returns the coefficients of the Newton interpolation polynomial (divided differences)
def getNewtonPolinom(ArgValues, FunctionValues):
    n = len(ArgValues)
    DivDiffs = []
    for k in range(n):
        F = getF(0, k, ArgValues, FunctionValues)
        DivDiffs.append(F)
    return DivDiffs

# Returns the value of Newton's polynomial at point t
def getNewtonValue(t, ArgValues, NewtonPol):
    res = 0
    n = len(ArgValues)
    for k in range(n):
        Mult = 1
        for i in range(k):
            Mult *= (t - ArgValues[i])
        res += Mult * NewtonPol[k]
    return res

# Returns an array of Newton polynomial values ​​for each Args value
# ArgValues ​​- array of n values ​​of argument t -- [t0, ... tn]
# FunctionValues ​​- an array of n function values ​​at points [t0, ... tn]
def getNewtonValues(Args, ArgValues, FunctionValues):
    NewtonValues = []
    NewtonPol = getNewtonPolinom(ArgValues, FunctionValues)
    for Arg in Args:
        NewtonValues.append(getNewtonValue(Arg, ArgValues, NewtonPol))
    return NewtonValues

NewtonArgValues      = np.arange(0, 1.1, 0.1)
NewtonFunctionValues = [1, 0.8, 0.5, 0.307, 0.2, 0.137, 0.1, 0.075, 0.06, 0.047, 0.039]
    
def main():
    plt.figure(figsize = (5, 5))
    plt.title("Newton's interpolation polynomial")

    Args = np.arange(0, 1.01, 0.01)
    NewtonVals = getNewtonValues(Args, NewtonArgValues, NewtonFunctionValues)
    plt.plot(Args, NewtonVals, 'b')

    plt.scatter(NewtonArgValues, NewtonFunctionValues, marker = "^")

    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

