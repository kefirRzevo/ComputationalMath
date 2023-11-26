import numpy as np
import math
import scipy

from scipy import integrate
import matplotlib.pyplot as plt
from numpy.linalg import norm

import task1

def Function2(X : float) -> float:
    return math.cos(X) / (2 + X * X)

def main():
    Epsilon = 5 * 1e-5
    N = 100000
    a = 0
    b = 20000
    Exact = scipy.integrate.quad(Function2, a, b)
    Simpson = task1.calculateSimpson(Function2, a, b, N)
    while (abs(Exact[0] - Simpson) > Epsilon):
        N *= 2
        Simpson = task1.calculateSimpson(Function2, a, b, N)
    print("\n\nSimpson method I =", Simpson)
    print("\n\nExact solution and its error (I, delta I) =", Exact)
    print("\n\nDifference from exact: |delta I| =", abs(Simpson - Exact[0]))
    print("\nIntegration step estimate: h =", 1.0 * (b - a) / N)

if __name__ == "__main__":
    main()
