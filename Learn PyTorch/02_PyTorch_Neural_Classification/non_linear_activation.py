import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_function import plot_predictions, plot_decision_boundary

# Create a tensor
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(A.dtype)
print(A)

# Visualize the tensor
plt.plot(A)
plt.show()

plt.plot(torch.relu(A))
plt.show()

print(A)
