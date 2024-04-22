import numpy as np
import matplotlib.pyplot as plt

# parameters
L = 10  # length of the rod in meters
D = 1  # thermal diffusivity
t_max = 10  # simulation time
N = 100  # number of spatial grid points
M = 1000  # number of time grid points
dx = L / (N - 1)  # spatial step size
dt = t_max / (M - 1)  # time step size
r = D * dt / (dx ** 2)  # stability parameter

# initial conditions
x = np.linspace(-5, 5, N)
u_init = np.exp(-x ** 2)
u = np.zeros((M, N))
u[0, :] = u_init

# boundary conditions
u[:, 0] = 0  # Dirichlet boundary condition
u[:, -1] = 0  # Dirichlet boundary condition

# Crank-Nicolson method
A = np.zeros((N - 2, N - 2))
for i in range(N - 2):
    A[i, i] = 2 * (1 + r)
    if i > 0:
        A[i, i - 1] = -r
    if i < N - 3:
        A[i, i + 1] = -r
B = np.linalg.inv(A)
d = np.zeros(N - 2)
for n in range(1, M):
    d[0] = u[n - 1, 1]
    d[-1] = u[n - 1, -2]
    d[1:-1] = u[n - 1, 2:-1] + r / 2 * (u[n - 1, 1:-2] - 2 * u[n - 1, 2:-1] + u[n - 1, 3:])
    u[n, 1:-1] = B.dot(d)

# plot
X, T = np.meshgrid(x, np.linspace(0, t_max, M))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, u)
ax.set_xlabel('x - m')
ax.set_ylabel('t - s')
ax.set_zlabel('u - deg C')
plt.show()
