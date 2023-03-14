
import torch
from torch import nn # nn is building blocks for neural networks
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test,  y_test  = X[train_split:], y[train_split:]

# moodel_0
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


# Create the loss function
loss_fn = nn.L1Loss()

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# An epoch is one loop through the data...
epochs =  1

## training loop

for epochs in range(epochs):

    # Set the model to training mode
    model_0.train() # start tracking gradient 

    # 1. Forward pass 
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero the optimizer gradients
    optimizer.zero_grad()

    # 4. Perform backprogpagation 
    loss.backward()

    # 5. Step the optimizer 
    optimizer.step()

    model_0.eval() # turns off gradient tracking

    print(model_0.state_dict())