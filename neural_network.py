import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing


#Initialize Training Data and Targets
d = 20 #####INITIALIZE THE NUMBER OF DIMENSIONS#####

dataset = pd.read_csv('/Users/EricChen/PycharmProjects/COVIDModeling/data_100000_pts.csv')
dataset = dataset.to_numpy()
dataset = preprocessing.normalize(dataset)
target = dataset[:,0]
data = dataset[:,1:(d+1)]
data = torch.from_numpy(data).float()
target = torch.from_numpy(target)
num_pts = len(target) #Total number of points

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create random Tensors for weight and bias
w = torch.randn(d, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(1, device=device, dtype=dtype, requires_grad=True)

#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
optimizer = optim.Adadelta([w,b], lr=1)  #Learning rate

for i in range(10000):
    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    # Forward pass: compute predicted y using operations on Tensors (data@w is matrix/vector multiplication)
    output = torch.sigmoid(data@w + b)

    # Compute the loss using operations on Tensors.
    loss = torch.sum((output - target)**2)
    loss = loss / num_pts

    #Print iteration and loss
    print('Iter:%d, Loss:%.4f'%(i,loss.item()))

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    loss.backward()

    #Take a step of gradient descent
    optimizer.step()

print(target)
print(output)
