import matplotlib.pyplot as plt
import numpy as np
import math
import sympy as sp

# Returns the derivative of the given functions as a column vector (dt/dt, dy1/dt, dy2/dt)
def get_derivative(U: list) -> list:
    y1 = U[1]
    y2 = U[2]
    deriv = np.array([1, y1 - y1 * y2, -y2 + y1 * y2])
    return deriv

# Initial condition for the Cauchy problem in the format (t0, y1, y2)
def get_start_condition() -> list:
    return np.array([0, 2, 2])

# Solves the system of differential equations using the Adams method
def calculate_adams_trajectory(start_condition: list, start: float, end: float, step_size: float) -> list:
    trajectory = []
    k0 = start_condition
    k1 = start_condition
    k2 = start_condition
    num_steps = int((end - start) / step_size)
    for _ in range(num_steps):
        f2 = get_derivative(k2)
        f1 = get_derivative(k1)
        f0 = get_derivative(k0)
        k3 = k2 + step_size * (23.0/12 * f2 - 16.0/12 * f1 + 5.0/12 * f0)
        trajectory.append(k2)
        k0 = k1
        k1 = k2
        k2 = k3
    return trajectory

def main():
    # Set up the plot
    plt.figure(figsize=(8, 6))
    plt.title("Solving a system of differential equations using the Adams method:")

    # Define the parameters of the problem
    start = 0
    end = 10
    step_size = 0.1

    # Calculate the solution using the Adams method
    adams_values = calculate_adams_trajectory(get_start_condition(), start, end, step_size)

    # Extract the y1 and y2 values from the solution
    y1_values = []
    y2_values = []
    for value in adams_values:
        y1_values.append(value[1])
        y2_values.append(value[2])

    # Plot the solution
    plt.plot(np.linspace(start, end, len(y1_values)), y1_values, 'b', label="y1")
    plt.plot(np.linspace(start, end, len(y2_values)), y2_values, 'g', label="y2")

    # Add labels and show the plot
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
