import numpy as np
import math
import matplotlib.pyplot as plt

# Finds the Lagrange basis function lk(t)
# ArgValues ​​- array of n argument values ​​[t0, ... tn]
def getl(t, k, ArgValues):
    n = len(ArgValues)
    lk = 1
    for j in range(n):
        Denom = ArgValues[k] - ArgValues[j]
        lk *= ((t - ArgValues[j]) / Denom) if k != j else 1
    return lk

# Returns the value of the Lagrange polynomial at point t
def getLagrangePolinom(t, ArgValues, FunctionValues):
    n = len(ArgValues)
    Value = 0
    for k in range(n):
        Value += (getl(t, k, ArgValues) * FunctionValues[k])
    return Value

# Returns an array of Newton polynomial values ​​for each Args value
# ArgValues ​​- array of n values ​​of argument t -- [t0, ... tn]
# FunctionValues ​​- an array of n function values ​​at points [t0, ... tn]
def getLagrangeValues(Args, ArgValues, FunctionValues):
    Values = []
    for Arg in Args:
        Values.append(getLagrangePolinom(Arg, ArgValues, FunctionValues))
    return Values

LagrangeArgValues      = np.arange(-0.8, 1.2, 0.2)
LagrangeFunctionValues = [0.02, 0.079, 0.175, 0.303, 0.459, 0.638, 0.831, 1.03, 1.23, 1.42]

def main():
    plt.figure(figsize = (5, 5))
    plt.title("Lagrange interpolation polynomial")

    Args = np.arange(-0.8, 1.01, 0.01)
    LagrangeVals = getLagrangeValues(Args, LagrangeArgValues, LagrangeFunctionValues)
    plt.plot(Args, LagrangeVals, 'b')

    plt.scatter(LagrangeArgValues, LagrangeFunctionValues, marker = "^")
    
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    main()

