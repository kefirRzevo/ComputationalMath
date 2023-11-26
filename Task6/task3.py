import numpy as np
import math
import scipy

from scipy import integrate
import matplotlib.pyplot as plt
from numpy.linalg import norm

import task1

def Function3(X : float) -> float:
    return math.sin(X) / math.sqrt(X)

def main():
    N = 10000
    a = 0.001
    b = 10
    Trapezoid = task1.calculateTrapezoid(Function3, a, b, N)
    Exact = scipy.integrate.quad(Function3, a, b)
    print("\n\nTrapezoid method I =", Trapezoid)
    print("\n\nExact solution and its error (I, delta I) =", Exact)
    print("\n\nDifference from exact: |delta I| =", abs(Trapezoid - Exact[0]))

if __name__ == "__main__":
    main()
