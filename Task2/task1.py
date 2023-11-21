import math
import numpy as np
import matplotlib.pyplot as plt

T = 0.5
T0 = 0
dT = 0.001
#machine error
Em = 10**(-15)

def u(t):
    return math.sin(t)

def diffUNumerically(n):
    M0 = abs(u(T0))
    M0h = abs(u(T0 + 0.01 * n))
    hopt = 0
    if (M0 > M0h):
        hopt = hOptimal(n, M0)
    else:
        hopt = hOptimal(n, M0h)
    Result = 0
    if (n == 1):
        return (u(T0) - u(T0 - hopt))/hopt
    for num in range(n + 1):
        k = -1 * (num - 1)
        uCurr = u(T0 + k * hopt)
        C = math.factorial(n) / (math.factorial(num) * math.factorial(n - num))
        Result += C * uCurr * (-1)**num
    Result /= hopt**n
    return Result

def diffSinExact(n):
    return math.sin(T0 + math.pi * n / 2)

def hOptimal(n, M0):
    if (M0 < 0):
        M0 *= (-1)
    if (n == 0):
        return 0.01
    Diff = diffSinExact(n + 1)
    if (Diff < 0):
        Diff *= (-1)
    isZero = 0
    if (Diff < Em):
        isZero = 1
        Diff = diffSinExact(n + 2)
        if (Diff < 0):
            Diff *= (-1)
    if (isZero):
        hOptim = Em**(1. / (n + 1)) * (math.factorial(n + 1) * M0 / (Diff * T**(n+1)))**(1. / (n + 1))
    hOptim = Em**(1. / n) * (math.factorial(n) * M0 / (Diff * T**n))**(1. / n)
    if (hOptim > 0.1):
        hOptim = 0.01
    print(f"For n = {n:2} step is {hOptim:.4}")
    return hOptim

def uMaccloren(un, n, diffN):
    return un + (diffN * T**n) / math.factorial(n)

def derivatives(n):
    Diffs = np.empty(n + 1)
    Diffs[0] = u(T0)
    for i in range(1, n):
        Diffs[i] = diffUNumerically(i)
    return Diffs

N = 15
Diffs = derivatives(N)
Functions = np.empty(N)
Functions[0] = u(T0)
Epsilons = np.empty(N)

for i in range(1, N):
    Functions[i] = uMaccloren(Functions[i - 1], i, Diffs[i])

for i in range(1, N):
    Func = Functions[i]
    Func0 = math.sin(T)
    Epsilons[i] = abs(Func - Func0)
    print(f"n = {i:2}; function {Functions[i]:.4}; error {Epsilons[i]:.4}")

X = [i for i in range(1, N)]

plt.figure(100)
plt.title("ln|u - u*|(n) in Macloren row")

plt.xlabel('n')
plt.ylabel('ln(|u - u*|)')

Y = [math.log(Epsilons[i]) for i in range (1, N)]
Pol = np.polyfit(X, Y, 3)
Ylin = [Pol[0] * x**3 + Pol[1] * x**2 + Pol[2] * x + Pol[3] for x in X]

plt.scatter(X, Y)
plt.plot(X, Ylin, 'b')

print(f"Error for n in range from {3} to {12} is less, than 0.01")
plt.show()
