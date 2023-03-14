import torch
from torch import nn # nn is building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test,  y_test  = X[train_split:], y[train_split:]

# plot graph
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    
    """ Plots training data, test data and compares predictions. """
    
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14});
    plt.show()

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias

torch.manual_seed(42)

model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(X_test)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epoch_count         = []
train_loss_values   = []
test_loss_values    = []

epochs =  300

### training loop

for epoch in range(epochs):

    model_0.train() 

    y_pred = model_0(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

### Testing loop

    model_0.eval() 

    with torch.inference_mode(): 

        test_pred = model_0(X_test) 
        
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())

            # print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
            # print(model_0.state_dict(),"\n")

if True == False:
    # Plot the loss curves
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(prop={"size": 14})
    plt.show()           

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

with torch.inference_mode():
    y_preds_new = model_0(X_test)

# plot_predictions(predictions=y_preds)
# plot_predictions(predictions=y_preds_new)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH) 

# Check the saved file path with os
file_path = 'models/01_pytorch_workflow_model_0.pth'
print(f"Model file exists at: {os.path.abspath(file_path)}" if os.path.isfile(file_path) 
      else f"Model file does not exist at: {os.path.abspath(file_path)}")

loaded_model_0 = LinearRegressionModel()
status = loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(status)

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

print(y_preds_new == loaded_model_preds)