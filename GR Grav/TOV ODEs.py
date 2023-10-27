import matplotlib.pyplot as plt
import numpy as np
# Here I use eulers method. Can also use odeint from scipy to do these calculations.
import pandas as pd
# Everythings in CGI units

csv_file = '/Users/dixshant/Documents/OLD MAC/UofT/4th year/PHY478/EMN450_N2LO_MBPT3_beta_equilibrium_eft_bands_00d75_004d000_maxpc2-8e11_qrkagn_05d000_00d010/DRAWmod1000-000000/eos-draw-000000.csv'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Extract data from two columns and convert to a NumPy array
rho = df['baryon_density'].to_numpy()
pressure = df['pressurec2'].to_numpy()

print(rho)
print(pressure)

def f(d, x,y,z):
    """

    :param c: constant
    :param x: Radius r
    :param y: Mass
    :param z: Density
    :return: dy/dx
    """
    return d*(z)*(x)**2

def g(G,k,c2,d,x,y,z,n):
    """

    :param d: constant
    :param x: Radius r
    :param y: Mass
    :param z: Density
    :param n: EOS parameter
    :return: dz/dx
    """
    first = ((G*y)/(n * k * (x ** 2) * (z ** (n - 2))))
    second = (1 + (k * z ** (n - 1)) / c2)
    third = (1 + (d * k * (z ** n) * (x ** 3)) / (y * c2))
    fourth = (1 - 2 * G * y / (x * c2))**(-1)
    # print(f'first: {first}')
    # print(f'second: {second}')
    # print(f'third: {third}')
    # print(f'fourth: {fourth}')
    return - first * second * third * fourth
    # return -((G*y)/(n * k * (x ** 2) * (z ** (n - 2)))) * (1 + (k * z ** (n - 1)) / c2) * (1 + (d * k * (z ** n) * x ** 3) / y * c2) *(1 - 2 * G * y / (x * c2))**(-1)

# Initial conditions
z0 = 7e16
x_max = 0.5e6

n = 5/3
delta_x = 100
d = 4*np.pi
G= 6.7e-8
k = 1e10
c2 = 9*10**20
z1 = n*c2*z0/(d*G*z0+n*c2) # Calculated analytically to prevent numerical instability at x=0 for z1
z2 = z1 - ((d*G*z1**2)/(n*c2))
# print(f'z1: {z1}')
# print(f'z2: {z2}')
x_values = [0, 1, 2]
y_values = [0, 0, d*z1]
z_values = [z0, z1, z2]

# set the initial points
x = 2
y = d*z1
z = z2

while x <= x_max:

    # Calculate the derivatives
    dydx = f(d, x, y, z)
    dzdx = g(G,k,c2,d,x,y,z,n)

    # Update y and z using Euler's method
    y += dydx * delta_x
    # print(f"dydx: {'{:.2e}'.format(dydx)}")
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

# for i in range(len(x_values)):
#     print(f"x = {x_values[i]}, y = {y_values[i]}, z = {z_values[i]}")

# print(y_values)
Y = np.array(y_values)
X = np.array(x_values)/100000
# print(Y)
# print(X)

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


