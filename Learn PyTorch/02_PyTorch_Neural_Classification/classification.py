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
# print(f"First 5 samples of X:\n {X[:5]}")
# print(f"First 5 samples of y:\n {y[:5]}")

circles = pd.DataFrame({"X1" : X[:, 0],
                        "X2" : X[:, 1],
                        "label" : y})
# print(circles.head(10),"\n")

# Check different labels
# print(circles.label.value_counts())

import matplotlib.pyplot as plt
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

# 1. Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=5) 
    self.layer_2 = nn.Linear(in_features=5, out_features=1)

  # 3. Define a forward() method that outlines the forward pass
  def forward(self, x):
    return self.layer_2(self.layer_1(x)) # x -> layer_1 ->  layer_2 -> output
    
# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
print(model_0)

next(model_0.parameters()).device

# Let's replicate the model above using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# print(model_0.state_dict(),"\n")

# Make predictions
with torch.inference_mode():
  untrained_preds = model_0(X_test.to(device))
  # print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
  # print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
  # print(f"\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}")
  # print(f"\nFirst 10 labels:\n{y_test[:10]}")

# Setup loss and optimizer function

# Loss_fn = nn.BCELoss # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc

# View the first 5 outputs of the forward pass on the test data
model_0.eval() 
with torch.inference_mode():
  y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

# Use the sigmoid activation function on our model logits to turn them into prediction
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

print(torch.round(y_pred_probs))