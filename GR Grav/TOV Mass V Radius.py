import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Polytropic index = 5/3
#
Mass = np.array([2e30, 2.64e30, 2.85e30, 3e30, 2.85e30, 2.5e30, 1.9e30, 1.475e30, 5.3e29, 1.73e29, 5.5e28, 1.74e28])
radius = np.array([5, 8.5, 9.7, 13.4,16.3, 19.5, 25, 28, 42, 63, 93, 136])
labels = ['3e16', '1e16', '6e15','2e15','1e15', '5e14','2e14','1e14', '1e13', '1e12', '1e11','1e10']

# params, covariance = curve_fit(exponential_func, radius, Mass)
# a_fit, b_fit, c_fit, d_fit = params
# print(b_fit)
# y_fit = exponential_func(radius, a_fit, b_fit, c_fit, d_fit)
# plt.plot(radius, y_fit)

plt.errorbar(radius, Mass, fmt='o')
for i, label in enumerate(labels):
    plt.text(radius[i], Mass[i], label, fontsize=9, ha='center', va='bottom')
plt.title("Mass vs Radius, labels on points represents critical density")
plt.xlabel("Radius (Km)")
plt.ylabel("Ln[Mass] (Kg)")
plt.show()
#

# Polytropic index 4/3


#
# radius = np.log(np.array([0.00175, 0.0175, 0.09, 0.2, 0.45, 0.8, 1.7, 5]))
# Mass = np.log(np.ones(8)*2.5e23)
# labels = ['1e20','1e17', '1e15', '1e14', '1e13', '1e12', '1e11', '1e10']
#
# # params, covariance = curve_fit(exponential_func, radius, Mass)
# # a_fit, b_fit, c_fit, d_fit = params
# # print(b_fit)
# # y_fit = exponential_func(radius, a_fit, b_fit, c_fit, d_fit)
# # plt.plot(radius, y_fit)
#
# plt.errorbar(radius, Mass, fmt='o')
# # for i, label in enumerate(labels):
# #     plt.text(radius[i], Mass[i], label, fontsize=9, ha='left', va='bottom')
# plt.title("Mass vs Radius for cirtical density")
# plt.xlabel("Radius (Km)")
# plt.ylabel("Ln[Mass] (Kg)")
# plt.show()

