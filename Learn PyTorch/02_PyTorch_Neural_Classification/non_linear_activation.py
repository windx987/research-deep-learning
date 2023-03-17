import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
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

