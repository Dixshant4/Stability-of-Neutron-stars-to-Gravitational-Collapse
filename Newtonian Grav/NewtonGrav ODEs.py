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
z0 = 1e13
x_max = 5e4

n = 4/3
delta_x = 10
c = 4*np.pi
d= -6.7e-8/(n*(1e10))

x_values = [0, 1]
y_values = [0, 0]
z_values = [z0, z0]

# set the initial points
x = 1
y = 0
z = z0

while x <= x_max:

    # Calculate the derivatives
    dydx = f(c, x, y, z)
    dzdx = g(d, x, y, z, n)

    # Update y and z using Euler's method
    y += dydx * delta_x
    print(f"dydx: {dydx}")
    # print(f"y: {y}")
    # print(f"dzdx: {dzdx}")
    # print(f"z: {z}")
    z += dzdx * delta_x
    # print(f"z: {z}")

    # Update x
    x += delta_x

    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

# Print or use the results as needed
# for i in range(len(x_values)):
#     print(f"x = {x_values[i]}, y = {y_values[i]}, z = {z_values[i]}")

print(y_values)
Y = np.array(y_values)
X = np.array(x_values)/100000
print(Y)
print(X)

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(X, Y/(1000))
plt.title("mass vs radius")
plt.xlabel("Radius (Km)")
plt.ylabel("Mass (Kg)")

plt.subplot(2, 1, 2)
plt.plot(X, z_values)
plt.title("density vs radius")
plt.xlabel("Radius (Km)")
plt.ylabel("rho (g/cm^3)")
plt.tight_layout()

plt.show()


