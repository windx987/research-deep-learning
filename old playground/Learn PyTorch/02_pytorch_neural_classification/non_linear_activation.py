import torch
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from helper_function import plot_predictions, plot_decision_boundary

# Replicating non-linear activation functions

# Create a tensor
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(A, A.dtype)

# plt.plot(A)
# plt.show()

def relu(x: torch.tensor) -> torch.tensor:
    return torch.maximum(torch.tensor(0), x)

print(relu(A))

plt.plot(torch.relu(A))
plt.show()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

print(sigmoid(A))

plt.plot(sigmoid(A));
plt.show()

plt.plot(torch.sigmoid(A)); 
plt.show()

tensor_A = torch.arange(-100, 100, 1)
plt.plot(tensor_A)
plt.show()

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

plt.plot(tanh(tensor_A))
plt.show()

plt.plot(torch.tanh(tensor_A))
plt.show()

# https://youtube.com/watch?v=Fu273ovPBmQ
# https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions