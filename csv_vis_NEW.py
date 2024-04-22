import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Temperature Distribution data from the CSV file using the loadtxt() function from NumPy
data = np.loadtxt("NN_temperature_evolution_FC_may.csv", delimiter=",") #correct data
#data = np.loadtxt("NN_temperature_evolution_fc_new.csv", delimiter=",")

# Reshape the data into a 2D grid of X and Y values (Position & Time) using meshgrid() function.
# np.arange() creates a 1D array of values from 0 to the number of columns or rows in data.
# meshgrid() uses these arrays to create a grid of X and Y coordinates.
X, Y = np.meshgrid(np.arange(data.shape[1]), np.linspace(0, 5, data.shape[0]))

# Create a 3D plot
fig = plt.figure()

# Add a new 3D subplot to the figure using Matplotlib's add_subplot() function.
# The 111 argument specifies that there is only one subplot.
# The projection='3d' argument specifies that it should be a 3D plot.
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the data as a 3D surface plot using Matplotlib's plot_surface() function.
# X, Y, and data are the X, Y, and Z data to be plotted.
ax.plot_surface(X, Y, data)

# Classify the 3D plot axes
ax.set_xlabel('Position, x')
ax.set_ylabel('Time, t')
ax.set_zlabel('Temperature, U')

# Visualise the plot
plt.show()

# Create a new 2D plot
fig2, ax2 = plt.subplots()

# Loop through the data and plot temperature distribution for various time points
for i in range(0, data.shape[0], 4):  # Change the step size to control the number of time points to be plotted
    ax2.plot(X[0], data[i], label=f'Time={Y[i, 0]:.2f}')

# Classify the 2D plot axes
ax2.set_xlabel('Position, x')
ax2.set_ylabel('Temperature, U')

# Add a legend to the plot
ax2.legend()

# Visualise the 2D plot
plt.show()
