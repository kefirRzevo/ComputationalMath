import numpy as np
import math
import matplotlib.pyplot as plt

import task1
import task2

def main():
    ArgValuesT     = task1.NewtonArgValues
    FunctionValues = task2.getLagrangeValues(ArgValuesT, task2.LagrangeArgValues, task2.LagrangeFunctionValues)
    ArgValues      = task1.NewtonFunctionValues
    
    plt.figure(figsize = (5, 5))
    plt.title("Interpolation polynomials y = y(x)")
    Args = np.arange(0.459, 1.43, 0.01)
    FuncValsL = task2.getLagrangeValues(Args, FunctionValues, ArgValues)
    FuncValsN = task1.getNewtonValues(Args, FunctionValues, ArgValues)
    plt.plot(FuncValsL, Args, 'g', label = "Lagrange")
    plt.plot(FuncValsN, Args, 'b', label = "Newton")
    plt.scatter(ArgValues, FunctionValues, marker = "^")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize = (6, 6))
    plt.title("Difference between Newton and Lagrange interpolation polynomials")
    Diff = []
    for i in range(len(FuncValsL)):
        Diff.append(FuncValsN[i] - FuncValsL[i])
    plt.plot(Args, Diff, 'b')
    plt.grid()
    plt.show()

    x = 0.431
    delta = 0.01
    # Find the solution to the equation x(y) = 0.431
    Y =np.arange(0.63, 0.73, 0.01)
    print(getLagrangeValues(Y, FunctionValues, ArgValues))
    
    # Found
    y = 0.665
    
    F = getLagrangeValues([y + delta / 2.0, y - delta / 2.0], FunctionValues, ArgValues)
    # Derivative y'x = 1 / x'y
    Deriv = delta / (F[0] - F[1])
    print("Derivative at point x =", x, ": y'x =", Deriv)

    
if __name__ == '__main__':
    main()

