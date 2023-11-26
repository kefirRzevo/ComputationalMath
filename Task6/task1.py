import numpy as np
import math
import scipy

from scipy import integrate
import matplotlib.pyplot as plt
from numpy.linalg import norm

def Function(X : float) -> float:
    return math.sin(100.0 * X) * math.exp(-X * X) * math.cos(2.0 * X)

# Returns one iteration of the Simpson method
def getOneSimpson(h : float, F, X1 : float, X2 : float, X3 : float) -> float:
    return h / 3.0 * (F(X1) + 4.0 * F(X2) + F(X3))

# Calculate the integral of F on the interval [a, b] using the Simpson method over N points
def calculateSimpson(F, a : float, b : float, N : int) -> float:
    Int = 0
    k = int(N / 2 - 1)
    h = 2.0 * (b - a) / N
    for i in range(k):
        X1 = a + 2 * i * h
        X2 = X1 + h
        X3 = X2 + h
        Int += getOneSimpson(h, F, X1, X2, X3)
    return Int

# Returns one iteration of the trapezoid method
def getOneTrapezoid(h : float, F, X1 : float, X2 : float) -> float:
    return h / 2.0 * (F(X1) + F(X2))

# Calculate the integral of F on the segment [a, b] using the trapezoidal method over N points
def calculateTrapezoid(F, a : float, b : float, N : int) -> float:
    Int = 0
    h = 1.0 * (b - a) / N
    for i in range(N - 1):
        X1 = a + i * h
        X2 = X1 + h
        Int += getOneTrapezoid(h, F, X1, X2)
    return Int

def main():
    N=10000
    a = 0
    b = 3
    Simpson = calculateSimpson(Function, a, b, N)
    Trapezoid = calculateTrapezoid(Function, a, b, N)
    Exact = scipy.integrate.quad(Function, a, b)
    print("\n\nSimpson Method I =", format(Simpson, '.10f'))
    print("\n\nTrapezoid method I =", format(Trapezoid, '.10f'))
    print("\n\nExact solution and its error (I, delta I) =", Exact)
    print("\n\nDifference from the exact one:\n")
    print("1. Simpson |delta I| =", abs(Simpson - Exact[0]), "\n")
    print("2. Trapezoid |delta I| =", abs(Trapezoid - Exact[0 ]))
    
if __name__ == "__main__":
    main()
