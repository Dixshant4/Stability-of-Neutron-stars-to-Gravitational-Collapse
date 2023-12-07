from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# n = 1
def odes(x, r):
    # constants
    c = 4*np.pi
    d = -7.4e-28

    # assign each ODE to a vector
    M = x[0]
    D = x[1]

    # define each ODE
    dMdr = c*(D)*(r)**2
    dDdr = d*M/(((r)**2)*(D)**(1-2))

    return [dMdr, dDdr]

# initial conditions

x0 = [0, 10**10]

# define the step size

r = np.linspace(1, 1000, 1000)

x = odeint(odes,x0, r)

M = x[:,0]
D = x[:,1]

plt.plot(r, M)
plt.show()

plt.plot(r, D)
plt.show()
