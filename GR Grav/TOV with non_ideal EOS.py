import matplotlib.pyplot as plt
import numpy as np
# Here I use eulers method. Can also use odeint from scipy to do these calculations.
import pandas as pd
c2 = 9*10**20
csv_file0 = '/Users/dixshant/Documents/OLD MAC/UofT/4th year/PHY478/TOV/EMN450_N2LO_MBPT3_beta_equilibrium_eft_bands_00d75_004d000_maxpc2-8e11_qrkagn_05d000_00d010/DRAWmod1000-000000/eos-draw-000051.csv'
base_path = '/Users/dixshant/Documents/OLD MAC/UofT/4th year/PHY478/TOV/EMN450_N2LO_MBPT3_beta_equilibrium_eft_bands_00d75_004d000_maxpc2-8e11_qrkagn_05d000_00d010/DRAWmod1000-000000/eos-draw-'
# for i in range(100):
#     csv_file1 = f"{base_path}{i:06d}.csv"
#     # Read the CSV file using pandas
#     df = pd.read_csv(csv_file1)
#
#     # Extract data from two columns and convert to a NumPy array
#     rho = df['energy_densityc2'].to_numpy()
#     pressure = df['pressurec2'].to_numpy()
#     plt.plot(pressure, rho)
#     # plt.xlim(0, 50)
#     plt.title("pressure vs rho")
#     plt.xlabel("pressur ")
#     plt.ylabel("rho ")
# plt.show()

# Read the CSV file using pandas
df = pd.read_csv(csv_file0)

# Extract data from two columns and convert to a NumPy array
rho = df['energy_densityc2'].to_numpy()
pressure = df['pressurec2'].to_numpy()
pressure = pressure*c2

pressure = np.array([float(item) for item in pressure])
base_path = "/Users/dixshant/Documents/OLD MAC/UofT/4th year/PHY478/TOV/EMN450_N2LO_MBPT3_beta_equilibrium_eft_bands_00d75_004d000_maxpc2-8e11_qrkagn_05d000_00d010/DRAWmod1000-000000/macro-draw-"

"""Buchdahl's Theorem"""
def upper_mass_limit(R):
    G = 6.7e-8
    c2 = 9 * 10 ** 20
    return 4*R*100000*c2/(9*G)

radius_range = np.arange(0, 50.5, 0.5)

for i in range(100):
    csv_file1 = f"{base_path}{i:06d}.csv"
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file1)
    # Extract data from two columns and convert to a NumPy array
    mass = df['M'].to_numpy()
    radius = df['R'].to_numpy()
    plt.plot(radius, mass)
    plt.xlim(0, 50)
    plt.ylim(0, 3)
    plt.title("Mass vs Radius")
    plt.xlabel("Radius (Km)")
    plt.ylabel("Mass (solar masses)")
plt.plot(radius_range, upper_mass_limit(radius_range)/(2*10**33), label="Buchdahl's limit")  # this is the criteria set my Buchdahl's theorem
plt.legend()
plt.show()

def f(d, x,y,z):
    """
    :param c: constant
    :param x: Radius r
    :param y: Mass
    :param z: Density
    :return: dy/dx
    """
    return d*(z)*(x)**2

def g(G,c2,d,x,y,z,p):
    """
    :param G: Gravitational constant
    :param c2: light speed squared
    :param d: 4*pi
    :param x: distance from center
    :param y: mass
    :param z: density
    :param p: pressure
    :return:
    """
    first = (G*y*z)/(x ** 2)
    second = (1 + p/(z*c2))
    third = (1 + (d  * p * (x ** 3)) /(y*c2) )
    fourth = (1 - (2 * G * y) / (x * c2))**(-1)
    return first * second * third * fourth

Mass = []
Radius = []

# In this for loop, I am integrating over the EOS
for i in range(len(pressure)):
    # Initial conditions
    # Make a plot for index 10, 140, 210, 310
    z0 = rho[len(rho)-(i*10 +1)]  # 1 data point for mass and radius for a given central pressure/density
    p0 = pressure[len(pressure)-(i*10 +1)]
    x_max = 1e7   # in cm

    delta_x = 100  # in cm
    d = 4*np.pi
    G= 6.7e-8
    c2 = 9*10**20
    m1 = d*z0/3

    x_values = [0, 1]
    y_values = [0, m1]
    z_values = [z0, z0]
    p_values = [p0, p0]


    # set the initial points
    x = 1
    y = m1
    z = z0
    p = p0

    while x <= x_max:

        # Calculate the derivatives
        dydx = f(d, x, y, z)
        dpdx = g(G,c2,d,x,y,z,p)
        # Update y and p using Euler's method
        y += dydx * delta_x
        p -= dpdx * delta_x
        z = np.interp(p, pressure, rho)
        # Update x
        x += delta_x
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        p_values.append(p)
    print(i)
    sign_change = None  # Initialize to None
    # Iterate through the data points
    # here i am finding the index at which pressure goes from pos to neg
    for a in range(1, len(p_values)):
        if p_values[a] < 0 and p_values[a - 1] > 0:
            sign_change = a
            break
        # print(x_values[sign_change]/100000)
    if sign_change is not None:
        Radius.append(x_values[sign_change]/100000)
        Mass.append(y_values[sign_change]/(2*10**33))
    else:
        plt.plot(np.array(Radius), upper_mass_limit(np.array(Radius)) / (2 * 10 ** 30))
        plt.plot(Radius, Mass)
        plt.title("Mass vs Radius")
        plt.xlabel("Radius (Km)")
        plt.ylabel("Mass (solar masses)")
        plt.show()
        break


# Y = np.array(y_values)
# X = np.array(x_values)/100000
#
# plt.figure(figsize=(8, 6))
# plt.subplot(3, 1, 1)
# plt.plot(X, Y/(1000))
# plt.title("mass vs radius")
# plt.xlabel("Radius (Km)")
# plt.ylabel("Mass (Kg)")
#
# plt.subplot(3, 1, 2)
# plt.plot(X, z_values)
# plt.title("energy density vs radius")
# plt.xlabel("Radius (Km)")
# plt.ylabel("rho (g/cm^3)")
#
# plt.subplot(3, 1, 3)
# plt.plot(X, p_values)
# plt.title("pressure vs radius")
# plt.xlabel("Radius (Km)")
# plt.ylabel("pressure (g/cm^3)*c2")
#
# plt.tight_layout()
#
# plt.show()