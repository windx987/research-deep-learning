## pytorch workflow
what_were_covering = {  1: "data (prepare and load)",
                        2: "build model",
                        3: "fitting the model to data (training)",
                        4: "making predictions and evaluating a model (inference)",
                        5: "saving and loading a model",
                        6: "putting it all together"
                        }

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path

# Machine learning is a game of two parts :
    # Get data into a nurical representation.
    # Build a model to learn patterns in that numerical representation.

## 1. Data < preparing and loading >

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]
# print(X, y)

## Split data into training and test sets 
# (one of the most important concepts in machine learning in general)

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# How might we better visualize our data? answer is plot data with matplotlib
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    
    """Plots training data, test data and compares predictions."""
    
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

# plot_predictions() #run functions plot 

## 2. Build model < Our first PyTorch model >

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)
    
# Checking the contents of a PyTorch model

#create random seed for tracking our model 
torch.manual_seed(42) # (weight & bias won't change all the time)  // why is 42. //ask google! 

#create an instance of the model 
model_0 = LinearRegressionModel() # subclass of nn.Module that contains nn.Parameter(s)

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(list(model_0.parameters()))

# We can also get the state (what the model contains) of the model using .state_dict()

# List named parameters 
print(model_0.state_dict())

# Making predictions using torch.inference_mode()
# torch.no_grad() do similar things but slower

# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

#run functions plot with predictions
# plot_predictions(predictions=y_preds)

## 3. Train model 

# Right now our model is making predictions using random parameters to make calculations, it's basically guessing (randomly).

## One way to measure how wrong your models predictions are is to use a loss function than optimize it.
# • Loss function: A function to measure how wrong your model's predictions are to the ideal outputs, lower is better.
# • Optimizer: Takes into account the loss of a model and adjusts the model's parameters (e.g. weight & bias)

# And specifically for PyTorch, we need: • A training loop  • A testing loop


# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss // Mean Absolute Error loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

# An epoch is one loop through the data...
epochs =  200

# Create empty loss lists to track values
epoch_count         = []
train_loss_values   = []
test_loss_values    = []

### Training loop
    # 0. Loop through the data
for epoch in range(epochs):

    # Set the model to training moade
    model_0.train() # train mode in pytorch, sets all parameters that require gradients to require gradients
    
    # 1. Forward pass 
    y_pred = model_0(X_train) # Pass the data through the model

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)  # Calculate the loss value

    # 3. Zero the optimizer gradients
    optimizer.zero_grad() 

    # 4. Perform backprogpagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()

    ### Testing loop
    model_0.eval() # Put the model in evaluation mode
    with torch.inference_mode(): # turn off gradients tracking & couple more things behind the scenes
        
        # 1. Forward pass 
        test_pred = model_0(X_test) # Calculate the loss value
        
        # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, y_test)


        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())

            # Print out what's happening
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

            #print out model status_dict()
            print(model_0.state_dict())

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

# Loading a saved PyTorch model's state_dict()

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH) # saves only the models learned parameters

