
## pytorch workflow
what_were_covering = {  1: "data (prepare and load)",
                        2: "build model",
                        3: "fitting the model to data (training)",
                        4: "making predictions and evaluating a model (inference)",
                        5: "saving and loading a model",
                        6: "putting it all together"
                        }



import torch
from torch import nn #nn contains all of Pytorch's building box for neural networks
import matplotlib.pyplot as plt

# Check Pytorch version
print(torch.__version__)

# 1. Data (preparing and loading)

# Data can be almost anything... in machine learning.
    # Excel speadsheet
    # Images of any kind
    # Videos (YouTube has lots of data... )
    # Audio like songs or podcasts
    # DNA
    # Text

# Machine learning is a game of two parts :
    # Get data into a nurical representation.
    # Build a model to learn patterns in that numerical representation.



# To showcase this, let's create some *known* data using the linear regression formula.
# We'll use a linear regression formula to make a straight line with known "parameters".

#Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10],"\n",len(X), len(y))

## Split data into training and test sets (one of the most important concepts in machine learning in general)
# Let's create a training and test set with our data.

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

print(len(X_train), len(y_train), len(X_test), len(y_test))

# How might we better visualize our data?
# This is where the data explorer's motto comes in!
# "Visualize, visualize, visualize!"

def plot_predictions(   train_data = X_train,
                        train_labels = y_train,
                        test_data=X_test,
                        test_labels=X_test,
                        predictions=None ):
   
    """Plots training data, test data and compares predictions."""
    
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")


    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    


    # Are there predictions?
    if predictions is not None:
        #Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

plot_predictions()

## 2. Build model < Our first PyTorch model >
