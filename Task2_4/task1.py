import numpy as np
import matplotlib.pyplot as plt
from time import sleep

h = 1
tau = 1
a = 0.5
u0 = np.zeros(31)

# u0[0] = 0
u0[10] = 1
u0[11] = 1
u0[12] = 1

def getNextTimeCutRightTriangle(uCurr):
    uList = []
    if(a < 0):
        for i in range(len(uCurr) - 1):
            uList.append((-abs(a * tau / h)) * (uCurr[i+1] - uCurr[i]) + uCurr[i])
        uList.append(0)
    if(a >= 0):
        uList.append(0)
        for i in range(1, len(uCurr) - 1):
            uList.append((-abs(a * tau / h)) * (uCurr[i] - uCurr[i-1]) + uCurr[i])
    return uList

def getNextTimeCutLax(uCurr):
    uList = []
    uList.append(0)
    for i in range(1, len(uCurr) - 1):
        uList.append((-a * tau / (2*h)) * (uCurr[i + 1] - uCurr[i - 1]) + 0.5 * (uCurr[i + 1] - uCurr[i - 1]))
    uList.append(0)
    return uList

def explicitLax(u0, tLast):
    numIt = int(tLast / tau)
    length = len(u0)
    u = [u0]
    for i in range(numIt):
        u.append(getNextTimeCutLax(u[i]))
    return u

def explicitRightTriangle(u0, tLast):
    numIt = int(tLast / tau)
    length = len(u0)
    u = [u0]
    for i in range(numIt):
        u.append(getNextTimeCutRightTriangle(u[i]))
    return u
    
def create_plot(x, y, title = ''):
    plt.figure(figsize = [12, 5])
    plt.stem(x, y)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)
    plt.grid()
    plt.show()

def main():
    u = explicitRightTriangle(u0, 20)
    for i in range(20):
        x = []
        _u = []
        for j in range(len(u[i])):
            x.append(j)
            _u.append(u[i][j])
        create_plot(x, _u, f'Right Triangle i = {i}')
    u = explicitLax(u0, 20)
    for i in range(20):
        x = []
        _u = []
        for j in range(len(u[i])):
            x.append(j)
            _u.append(u[i][j])
        create_plot(x, _u, f'Lax i = {i}')
    return 0

if __name__ == '__main__':
    main()
