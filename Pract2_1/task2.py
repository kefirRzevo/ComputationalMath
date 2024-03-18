import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp
from scipy. integrate import odeint

Mu = 0.00095388
Gamma = 1 - Mu
Epsilon = 1e-6
f = 0
R1 = [0.5 - 0.5 * Mu, -0.5 * math.sqrt(3)]
R2 = [0.5 * math.sqrt(3), 0.5 - 0.5 * Mu]
DeltaT = 0.1
Start = 0
Stop = 300


# Returns the derivative of the searched functions as a column
# (/*time derivative */ 1, x', Vx', y', Vy') from the task
def getDerivative(U : list) -> list:
    # Returns the distance, adjusted r1, to body 1
    def getDist1(X : float, Y : float):
        CoordX = X + Mu
        CoordY = Y
        return math.sqrt(CoordX**2 + CoordY**2)

    # Returns the r2-adjusted distance to body 2
    def getDist2(X : float, Y : float):
        CoordX = X - Gamma
        CoordY = Y
        return math.sqrt(CoordX**2 + CoordY**2)

    t, X, Vx, Y, Vy = U
    DerrVx = 2 * Vy + X - Gamma * (X + Mu) / getDist1(X, Y)**3 - \
        Mu * (X - Gamma) / getDist2(X, Y)**3 - f * Vx
    DerrVy = -2 * Vx + Y - Gamma * (Y) / getDist1(X, Y)**3 - \
        Mu * (Y ) / getDist2(X, Y)**3 - f * Vy
    Deriv = np.array([1, Vx, DerrVx, Vy, DerrVy])
    return Deriv

# Initial value of the Cauchy problem in the format (t0, x0, Vx0, y0, Vy0)
def getStartCoord() -> list:
    return np.array([0, R1[0], R1[1], R2[0], R2[1]])

# Depicts the trajectory X(t), Y(t) on the graph
def showTrajectory(Args : list, Vals : list, Title : str):
    plt.figure(figsize = (8, 6))
    plt.title(Title)

    X = []
    Y = []
    for Val in Vals:
        X.append(Val[1])
        Y.append(Val[3])
    if (len(Args) > len(X)):
        Args = np.delete(Args, 0, 0)
    plt.plot(Args, X, 'r', label = 'x')
    plt.plot(Args, Y, 'g', label = 'y')
    
    plt.legend()
    plt.grid()
    plt.show()

# Calculate trajectory using Adams method
def calculateAdamsTrajectory(StartCoord : list, Start: float, Stop: float, DeltaT: float) -> list:
    res_t = [Start, Start + DeltaT, Start + 2 * DeltaT]
    K0 = StartCoord
    K1 = K0 + DeltaT * getDerivative(K0)
    K2 = K1 + DeltaT * (3.0/2.0*getDerivative(K1) - 1.0/2.0*getDerivative(K0))
    res_y = [K0, K1, K2]
    res_f = [getDerivative(K0), getDerivative(K1), getDerivative(K2)]

    t = res_t[2]
    y = res_y[2]
    NumPoints = int((Stop - Start) / DeltaT)
    for i in range(2, NumPoints):
        y = y + DeltaT * (23.0/12 * res_f[i] - 16.0/12 * res_f[i-1] + 5.0/12 * res_f[i-2])
        t += DeltaT
        res_t.append(t)
        res_y.append(y)
        res_f.append(getDerivative(y))
    return res_y, res_t

def main():
    print('No adams')
    #AdamsVals, Args = calculateAdamsTrajectory(getStartCoord(), Start, Stop, DeltaT)
    #showTrajectory(Args, AdamsVals, "Solving a system of differential equations using the Adams method")

if __name__ == '__main__':
    main()

# Calculates the norm without taking into account the first time coordinate
def getNorm(U : list):
    t, X, Vx, Y, Vy = U
    return math.sqrt(X**2 + Vx**2 + Y**2 + Vy**2)

# Calculates the trajectory using the Runge-Kutta-Fehlberg method
def calculateRungeKuttaTrajectory(StartCoord : list, Start: float, Stop: float, DeltaT: float) -> list:
    Args = []
    Trajectory = []
    U0 = StartCoord
    NumPoints = int((Stop - Start) / DeltaT)
    for Point in range(NumPoints):
        Args.append(Start + Point * DeltaT)
        K1 = getDerivative(U0)
        K2 = getDerivative(U0 + DeltaT / 4 * K1)
        Point3 = U0 + 3 * DeltaT / 32 * K1 + 9 * DeltaT / 32 * K2
        Point3[0] = U0[0] + 3 * DeltaT / 8
        K3 = getDerivative(Point3)
        Point4 = U0 + 1932 * DeltaT / 2197 * K1 - 7200 * DeltaT / 2197 * K2 + 7296 * DeltaT / 2197 * K3
        Point4[0] = U0[0] + 12 * DeltaT / 13
        K4 = getDerivative(Point4)
        Point5 = U0 + 439 * DeltaT / 216 * K1 - 8 * DeltaT * K2 + 3680 * DeltaT / 513 * K3 - 845 * DeltaT / 4104 * K4
        Point5[0] = U0[0] + DeltaT
        K5 = getDerivative(Point5)
        Point6 = U0 - 8 * DeltaT / 27 * K1 + 2 * DeltaT * K2 - 3544 * DeltaT / 2565 * K3 + 1859 * DeltaT / 4104 * K4 - 11 * DeltaT / 40 * K5
        Point6[0] = U0[0] + DeltaT / 2
        K6 = getDerivative(Point6)

        U1 = U0 + DeltaT * (25 * K1 / 216 + 1408 * K3 / 2565 + 2197 * K4 / 4104 - K5 / 5)
        U2 = U0 + DeltaT * (16 * K1 / 135 + 6656 * K3 / 12825 + 28561 * K4 / 56430 - 9 * K5 / 50 + 2 * K6 / 55)
        
        DeltaOpt = DeltaT * (Epsilon * DeltaT / (2 * getNorm(U1 - U2)))**(1.0 / 4)

        U1 = U0 + DeltaOpt * (25 * K1 / 216 + 1408 * K3 / 2565 + 2197 * K4 / 4104 - K5 / 5)
        U0 = U1
        Trajectory.append(U0)
    return Trajectory, Args

def main():
    RungeKuttaVals, Args = calculateRungeKuttaTrajectory(getStartCoord(), Start, Stop, DeltaT)
    showTrajectory(Args, RungeKuttaVals, "Solving a differential equation by the Runge-Kutta-Fehlberg method")

if __name__ == '__main__':
    main()

X = []
Y = []
Vals, Args = calculateRungeKuttaTrajectory(getStartCoord(), Start, Stop, DeltaT)
for Val in Vals:
    X.append(Val[1])
    Y.append(Val[3])

plt.plot(X, Y, 'r', label = 'x(y)')
plt.legend()
plt.grid()
plt.show()

# import matplotlib.animation as animation
# import pandas as pd
# import os

# X_coord = []
# Y_coord = []

# def animate(i):
#     X_coord.append(df['x'].iloc[i])
#     Y_coord.append(df['y'].iloc[i])
#     ax.plot(X_coord,Y_coord, 'b')

# df = pd.DataFrame()
# Args = np.arange(Start, Stop, DeltaT)
# vals = calculateRungeKuttaTrajectory(getStartCoord(), Start, Stop, DeltaT)
# X = []
# Y = []
# for Val in vals:
#     X.append(Val[1])
#     Y.append(Val[3])
# df['x'] = X
# df['y'] = Y
# df.head()
# dataPath = os.path.join(os.path.dirname(__file__), 'dataset.csv')
# df.to_csv(dataPath, index = False)

# df = pd.DataFrame()
# df = pd.read_csv(dataPath)

# fig,ax = plt.subplots(figsize = (12,12))

# Lim = 50
# ax.set_xlim(-Lim, Lim)
# ax.set_ylim(-Lim, Lim)

# ax.set_title('Three body problem', fontsize = 18)

# #Yupiter = plt.Circle((R1[0], R1[1]), 0.3, color='r', fill=True)
# #Sun = plt.Circle((R2[0], R2[1]), 0.4, color='y', fill=True)

# ax = plt.gca()

# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))

# #ax.add_patch(Yupiter)
# #ax.add_patch(Sun)

# anim = animation.FuncAnimation(fig, animate,  frames = len(df.index), interval = len(df.index))
# gifPath = os.path.join(os.path.dirname(__file__), 'ThreeCosmicBodies.gif')
# anim.save(gifPath, fps = 50, writer = 'pillow')
