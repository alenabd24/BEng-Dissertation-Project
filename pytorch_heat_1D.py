#%%
import matplotlib.animation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import grad
import pandas as pd
import matplotlib.pyplot as plt
#!%matplotlib tk
import matplotlib.pyplot as plt

# Set up neural net (fully connected - 4 hidden layers)
# One input layer with two nodes, followed by four hidden layers with 20 nodes each.
# The output layer has one node, which represents the predicted temperature at a given time and location.
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.input = nn.Linear(2, 20)
      self.hidden = nn.Linear(20, 20)
      self.output = nn.Linear(20, 1)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden(x))
        output = self.output(x)

        return output


net = Net()

#Limits for heat equation
tlim = [0.,5.]             #list of 2 elements
xlim = [-5.,5.]            #list of 2 elements


#Define Loss Function
def loss_fn(net):

  # Creates two tensors (t,x) that are each 1D tensors with 50 elements.
  # The elements of t and x are evenly spaced values between tlim (0 to 5) and xlim(-5 to 5) respectively.
  # The requires_grad flag is set to True, so gradients will need to be computed with respect to these tensors during backpropagation.
  t = torch.linspace(tlim[0],tlim[1],50,requires_grad=True)     #size 1x50
  x = torch.linspace(xlim[0],xlim[1],50,requires_grad=True)     #size 1x50


  # This creates a grid of points (t[i], x[j]) for all pairs of i and j between 0 and 49, inclusive.
  # The resulting tensors tt and xx are each 2D tensors with 50 rows and 50 columns, representing the t and x values at each point on the grid.
  # The reshape function then flatten these tensors to have 2500 rows and 1 column.
  tt,xx = torch.meshgrid(t,x)    #both tensors have size 50x50
  tt = tt.reshape([-1,1])        #size 2500x1
  xx = xx.reshape([-1,1])        #size 2500x1

  # This reshapes the t and x tensors to be 2D tensors with 50 rows and 1 column, which is useful for later computations.
  t = t.reshape([-1,1]) #size 50x1
  x = x.reshape([-1,1]) #size 50x1

  # This creates two tensors t0 and x0 that are each 2D tensors with 50 rows and 1 column.
  # The elements of t0 are all equal to tlim[0], while the elements of x0 are the same as those in the x tensor.
  t0 = tlim[0]*torch.ones_like(x)       #size 50x1
  x0 = x                                #size 50x1

  # Vectors for boundary space and time point
  # This creates two tensors xb1 and xb2, each with 50 elements, and sets them to xlim[0] and xlim[1], respectively.
  # The requires_grad=True flag is set so that gradients can be computed with respect to these tensors during backpropagation.
  # The torch.cat function is used to concatenate xb1 and xb2 tensors vertically into a single 100-element tensor xb.
  # Similarly, t and t tensors are concatenated vertically into a single 100-element tensor tb.
  xb1 = xlim[0]*torch.ones_like(t,requires_grad=True)      #size 50x1
  xb2 = xlim[1]*torch.ones_like(t,requires_grad=True)      #size 50x1
  xb = torch.cat([xb1,xb2],0)                              #size 100x1
  tb = torch.cat([t,t],0)                                  #size 100x1 

  # Aggregate these points into Nx2 input tensor of (time,space) tuples
  # This creates three 2D tensors inputs, inputs0, and inputsb. inputs is a tensor of size 2500x2,
  # where the first column contains the elements of tt and the second column contains the elements of xx.
  # inputs0 is a tensor of size 50x2, where the first column contains the elements of t0 and the second column contains the elements of x0.
  # inputsb is a tensor of size 100x2, where the first column contains the elements of tb and the second column contains the elements of xb.
  inputs  = torch.cat([tt,xx],1)                           #size 2500x2
  inputs0 = torch.cat([t0,x0],1)                           #size 50x2
  inputsb  = torch.cat([tb,xb],1)                          #size 100x2


  #Evaluate network on these inputs
  u  = net.forward(inputs)
  u0 = net.forward(inputs0)
  ub = net.forward(inputsb)

  #Calculate differential operator terms
  u_t  = grad(u,tt,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]   #first order time derivative
  u_x  = grad(u,xx,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]   #first order spatial derivative
  u_xx = grad(u_x,xx,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0] #second order spatial derivative

  #Calculate boundary operator terms for Neumann boundary conditions
  #ub_x = grad(ub,xb,grad_outputs=torch.ones_like(ub),retain_graph=True,create_graph=True)[0]

  # Initial condition function (just a Gaussian)
  g = torch.exp(-0.5*x**2)

  # Using computed terms, construct loss function and return this
  res = (u_xx - u_t)**2 #residual of the differential equation
  # bc  = ub_x**2 #zero heat mass conservation for Neumann boundary conditions
  bc  = (ub-0)**2 #zero heat mass conservation (boundary condition)
  ic  = (u0 - g)**2 #initial condition

  # The loss function constructed as the sum of the means of the three defined vectors
  loss=torch.mean(res) + torch.mean(bc) + torch.mean(ic)

  return loss

# Training the neural network. The optimizer used for training is the Adam optimizer (torch.optim.Adam)
# Initialize empty lists to store loss and iteration values
stored_losses = []
stored_iterations = []

# Define the optimizer and learning rate
opt = torch.optim.Adam(net.parameters(), lr=0.002)

# Train the network for 2000 iterations using a learning rate of 0.002
for i in range(2000):
    opt.zero_grad()
    loss = loss_fn(net)
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print('Iteration:', i, '. Loss:', loss)

    # Append the loss and iteration values to the respective lists
    stored_losses.append(loss.item())
    stored_iterations.append(i)

# Redefine the optimizer with a smaller learning rate (0.0002) and continue training
opt = torch.optim.Adam(net.parameters(), lr=0.0002)

for i in range(2000):
    opt.zero_grad()
    loss = loss_fn(net)
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print('Iteration:', i, '. Loss:', loss)

    # Append the loss and iteration values to the respective lists
    stored_losses.append(loss.item())
    stored_iterations.append(i + 2000)  # Add 2000 to the iteration count for the second loop

# Plot the loss values against the number of iterations
plt.plot(stored_iterations, stored_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# CODE THAT SAVES THE TEMPERATURE DISTRIBUTION OF THE CONSIDERED DOMAIN OVER TIME
# The temperature distribution is saved in the XX tensor
# The tensor xx represents the space, and tt represents the time
xx = torch.linspace(xlim[0],xlim[1],50,requires_grad=True) 
tt = torch.linspace(tlim[0],tlim[1],50,requires_grad=True)

# The meshgrid function from torch is used to create a 2D grid of the space and time values
xx, tt = torch.meshgrid(tt,xx)   
xx = xx.reshape(-1,1)
tt  = tt.reshape(-1,1)

# The torch.hstack function is then used to concatenate the tt and xx tensors horizontally into a 2D tensor.
# The temperature values at each point in the grid are then calculated using the net.forward function.
XX = torch.hstack([tt,xx])
yy = net.forward(XX).detach().numpy()
yy=yy.reshape(50,50)
np.savetxt("NN_temperature_evolution.csv", yy, delimiter=",")

#net.forward(torch.Tensor([t,x]))     # Command for calling a temperature in any given time and space
