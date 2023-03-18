import torch
from torch import nn

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from helper_function import plot_predictions, plot_decision_boundary

from torchmetrics import Accuracy

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Make 1000 samples
n_samples = 1000

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create circles
X, y = make_moons(n_samples, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

moons = pd.DataFrame({"X1" : X[:, 0],
                        "X2" : X[:, 1],
                        "label" : y})
print(moons)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
class MoonModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=5)
        self.layer3 = nn.Linear(in_features=5, out_features=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

# Train the model
torch.cuda.manual_seed(42)
torch.manual_seed(42)

# Instantiate the model
model_5 = MoonModelV0().to(device)
print(model_5)    

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss() # MAE loss with regression data
optimizer = torch.optim.SGD(params=model_5.parameters(), 
                            lr=0.1)  


# What's coming out of our model?
model_5.eval() 
with torch.inference_mode():
  y_logits = model_5(X_test.to(device))[:5]
print(f"Logits:{y_logits}")
y_pred_probs = torch.sigmoid(y_logits)
print(f"Pred probs:{y_pred_probs}")
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_5(X_test.to(device)[:5])))
print(f"Pred labels:{y_pred_labels}")

