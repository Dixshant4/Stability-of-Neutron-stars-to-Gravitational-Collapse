import matplotlib.pyplot as plt
import numpy as np
# Here I use eulers method. Can also use odeint from scipy to do these calculations


def f(c, x,y,z):
    """

    :param c: constant
    :param x: Radius r
    :param y: Mass
    :param z: Density
    :return: dy/dx
    """
    return c*(z)*(x)**2

def g(d, x,y,z,n):
    """

    :param d: constant
    :param x: Radius r
    :param y: Mass
    :param z: Density
    :param n: EOS parameter
    :return: dz/dx
    """
    return d*y/(((x)**2)*(z)**(n-2))

# Initial conditions
z0 = 10**8
x_max = 1000

n = 3
delta_x = 1
c = 4*np.pi
d= -6.7e-11/((3*10**8)**2*n)

x_values = [0, 1]
y_values = [0, 0]
z_values = [z0, z0]

# set the initial points
x = x_values[1]
y = y_values[1]
z = z_values[1]

while x <= x_max:

    # Calculate the derivatives
    dydx = f(c, x, y, z)
    dzdx = g(d, x, y, z, n)

    # Update y and z using Euler's method
    y += dydx * delta_x
    z += dzdx * delta_x

    # Update x
    x += delta_x

    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

# Print or use the results as needed
for i in range(len(x_values)):
    print(f"x = {x_values[i]}, y = {y_values[i]}, z = {z_values[i]}")

plt.plot(x_values, y_values)
plt.title("mass vs radius")
plt.show()

plt.plot(x_values, z_values)
plt.title("density vs radius")
plt.show()
