import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import os

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data for training
# linear regression formula (y = weight * X + bias)

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# X[:10], y[:10]

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing set 
X_train, y_train = X[:train_split], y[:train_split]
X_test,  y_test  = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14});
    plt.show()

def loss_curves(epochs_count, trains_loss_values, tests_loss_values):
    # Plot the loss curves
    plt.plot(epochs_count, trains_loss_values, label="Train loss")
    plt.plot(epochs_count, tests_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(prop={"size": 14})
    plt.show()

class LinearRegressionModel(nn.Module): # use nn.Linear 
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer(x)

torch.manual_seed(42)

model_1 = LinearRegressionModel()

with torch.inference_mode():
    y_pred = model_1(X_test)

plot_predictions(predictions=y_pred.cpu())

# Set the model to use target device
model_1.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs =  300

epoch_count         = []
train_loss_values   = []
test_loss_values    = []

# Put data on the available device
X_train, X_test, y_train, y_test = map(lambda x: x.to(device), (X_train, X_test, y_train, y_test))

for epoch in range(epochs):

## training
    model_1.train() 
    y_preds = model_1(X_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
## testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

            epoch_count.append(epoch)
            train_loss_values.append(loss.cpu().detach().numpy())
            test_loss_values.append(test_loss.cpu().detach().numpy())

# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print("\nThe model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

loss_curves(epoch_count, train_loss_values, test_loss_values)
plot_predictions(predictions=test_pred.cpu())

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# Check the saved file path
print(f"Model file exists at: {os.path.abspath(MODEL_SAVE_PATH)}" if os.path.isfile(MODEL_SAVE_PATH) 
      else f"Model file does not exist at: {os.path.abspath(MODEL_SAVE_PATH)}")

loaded_model_1 = LinearRegressionModel()
status = loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_1.to(device)

print(loaded_model_1, status)
print(loaded_model_1.state_dict())

loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)

print(test_pred == loaded_model_preds)
# print(test_pred,"\n", loaded_model_preds)