import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Polytropic index = 5/3
#
# mass = [5.53e26, 1.75e28, 5.5e28, 1.75e29, 5.5e29, 1.75e30, 5.5e30, 5.5e31, 1.75e33, 5.5e35]
# radius = np.sort(np.array([429, 130, 92.5, 63, 43, 27, 19.95, 9.25, 2.9, 0.43]))
# Mass = np.sort(np.log(np.array(mass)), kind='quicksort')[::-1]
# labels = ['1e7','1e10', '1e11', '1e12', '1e13', '1e14', '1e15', '1e17', '1e20', '1e25']
#
# params, covariance = curve_fit(exponential_func, radius, Mass)
# a_fit, b_fit, c_fit, d_fit = params
# print(b_fit)
# y_fit = exponential_func(radius, a_fit, b_fit, c_fit, d_fit)
# # plt.plot(radius, y_fit)
#
# plt.errorbar(radius, Mass, fmt='o')
# for i, label in enumerate(labels):
#     plt.text(radius[i], Mass[i], label, fontsize=9, ha='center', va='bottom')
# plt.title("Mass vs Radius for cirtical density")
# plt.xlabel("Radius (Km)")
# plt.ylabel("Ln[Mass] (Kg)")
# plt.show()
#

# Polytropic index 4/3



radius = np.log(np.array([0.00175, 0.0175, 0.09, 0.2, 0.45, 0.8, 1.7, 5]))
Mass = np.log(np.ones(8)*2.5e23)
labels = ['1e20','1e17', '1e15', '1e14', '1e13', '1e12', '1e11', '1e10']

# params, covariance = curve_fit(exponential_func, radius, Mass)
# a_fit, b_fit, c_fit, d_fit = params
# print(b_fit)
# y_fit = exponential_func(radius, a_fit, b_fit, c_fit, d_fit)
# plt.plot(radius, y_fit)

plt.errorbar(radius, Mass, fmt='o')
# for i, label in enumerate(labels):
#     plt.text(radius[i], Mass[i], label, fontsize=9, ha='left', va='bottom')
plt.title("Mass vs Radius for cirtical density")
plt.xlabel("Radius (Km)")
plt.ylabel("Ln[Mass] (Kg)")
plt.show()

