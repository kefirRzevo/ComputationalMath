import numpy as np
import matplotlib.pyplot as plt
from time import sleep

h = 1
tau = 1
u0 = np.zeros(31)

# u0[0] = 0
u0[10] = 1
u0[11] = 1
u0[12] = 1

def create_plot(x, y, title = ''):
    plt.figure(figsize = [12, 5])
    plt.stem(x, y)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)
    plt.grid()
    plt.show()

def getNextTimeCutLaxWind(uCurr):
    uList = [0]
    for i in range(1, len(uCurr) - 1):
        uList.append(uCurr[i] - (1 * tau / (2 * h)) * (uCurr[i + 1] - uCurr[i - 1]) + ((tau ** 2)/(h)) * ((uCurr[i + 1] - 2 * uCurr[i] + uCurr[i - 1])/(h**2)))
    uList.append(0)
    return uList

def explicitLaxWendroff(u0, tLast):
    numIt = int(tLast / tau)
    length = len(u0)
    u = [u0]
    for i in range(numIt):
        u.append(getNextTimeCutLaxWind(u[i]))
    return u

# a = 1???
# Тупо сделал как написано, хотя это релаьно что-то не то. Если a = uCurr[i], то волна затухает
def getNextTimeCutLeftCorner(uCurr):
    uList = [0]
    for i in range(1, len(uCurr) - 1):
        uList.append((-(1 * tau / h)) * (uCurr[i] - uCurr[i-1]) + uCurr[i])
    uList.append(0)
    return uList

def explicitLeftCorner(u0, tLast):
    numIt = int(tLast / tau)
    length = len(u0)
    u = [u0]
    for i in range(numIt):
        u.append(getNextTimeCutLeftCorner(u[i]))
    return u

def main():
    u = explicitLeftCorner(u0, 20)
    for i in range(20):
        x = []
        _u = []
        for j in range(len(u[i])):
            x.append(j)
            _u.append(u[i][j])
        create_plot(x, _u, f'LeftCorner i = {i}')
    u = explicitLaxWendroff(u0, 20)
    for i in range(20):
        x = []
        _u = []
        for j in range(len(u[i])):
            x.append(j)
            _u.append(u[i][j])
        create_plot(x, _u, f'LaxWendroff i = {i}')
    return 0

if(__name__ == '__main__'):
    main()
