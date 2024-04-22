import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import matplotlib.patches as mpatches

# Load the data from the CSV files
validationdata = np.genfromtxt('validationdata.csv', delimiter=',')
NN_temperature_evolution_fc = np.genfromtxt('NN_temperature_evolution_fc.csv', delimiter=',')

# Match the file sizes
x = np.linspace(0, 1, NN_temperature_evolution_fc.shape[1])
y = np.linspace(0, 1, NN_temperature_evolution_fc.shape[0])
f = RectBivariateSpline(y, x, NN_temperature_evolution_fc, kx=1, ky=1)
x_new = np.linspace(0, 1, validationdata.shape[1])
y_new = np.linspace(0, 1, validationdata.shape[0])
NN_temperature_evolution_fc_resized = f(y_new, x_new)

# Create the 3D overlaying graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for the 3D plot
time, spatial_dimension = np.meshgrid(x_new, y_new)

# Plot the validation data with blue color
surf1 = ax.plot_surface(time, spatial_dimension, validationdata, color='blue', alpha=0.5)

# Plot the NN_temperature_evolution_fc_resized data with red color
surf2 = ax.plot_surface(time, spatial_dimension, NN_temperature_evolution_fc_resized, color='red', alpha=0.5)

# Add labels and title
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Temperature')
ax.set_title('3D Overlaying Graph of Temperature Distributions')
ax.set_zlim(0.965, 1)

# Create custom legends for the surface plots with blue and red colors
legend_elements = [mpatches.Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='Crank-Nicolson Solution'),
                   mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='k', label='Neural Network Solution')]

ax.legend(handles=legend_elements, loc='upper right')

# Save the 3D overlaying graph to a file
plt.savefig('3D_overlaying_temperature_distributions.png', dpi=300)

# Show the 3D overlaying graph
plt.show()
