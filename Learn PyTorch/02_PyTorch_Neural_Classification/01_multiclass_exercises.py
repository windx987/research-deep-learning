import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

from helper_function import plot_predictions, plot_decision_boundary

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()

X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

class SpiralModel(nn.Module): 
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(in_features=2, out_features=10)
    self.linear2 = nn.Linear(in_features=10, out_features=10)
    self.linear3 = nn.Linear(in_features=10, out_features=3)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))

# Instantiate model and send it to device
model_6 = SpiralModel().to(device)
print(model_6)

# Put data on the available device
X_train, X_test, y_train, y_test = map(lambda x: x.to(device), (X_train, X_test, y_train, y_test))

# Print out untrained model outputs
print("Logits:")
print(model_6(X_train)[:10])

print("Pred probs:")
print(torch.softmax(model_6(X_train)[:10], dim=1))

print("Pred labels:")
print(torch.softmax(model_6(X_train)[:10], dim=1).argmax(dim=1))

model_6.to(device)

acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)
print(acc_fn)

# Setup loss function and optimizer and accuracy
loss_fn = nn.CrossEntropyLoss() # use logits, true_value
optimizer = torch.optim.Adam(model_6.parameters(), lr=0.02)

# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put data on the available device
X_train, X_test, y_train, y_test = map(lambda x: x.to(device), (X_train, X_test, y_train, y_test))

epochs = 1000

for epoch in range(epochs):
    model_6.train()
    y_logits = model_6(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_6.eval()
    with torch.inference_mode():
      test_logits = model_6(X_test)
      test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
      
      test_loss = loss_fn(test_logits, y_test)
      test_acc = acc_fn(test_preds, y_test)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_6, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_6, X_test, y_test)
plt.show()