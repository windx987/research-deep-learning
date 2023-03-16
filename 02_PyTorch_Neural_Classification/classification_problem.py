import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 1. Make classification data and get it ready

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# print(len(X), len(y))
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

circles = pd.DataFrame({"X1" : X[:, 0],
                        "X2" : X[:, 1],
                        "label" : y})
print(circles.head(10),"\n")

# Check different labels
# print(circles.label.value_counts())

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y, 
            cmap=plt.cm.RdYlBu);
# plt.show()

# print(X.shape, y.shape)
# print(X)

X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# turn data into tensors

print(type(X))

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5], y[:5])

print(type(X), X.dtype, y.dtype)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test), n_samples)

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print(X_train)

#page 66