import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # give the clusters a little shake up
                            random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)
# 4. Plot data (visualize, visualize, visualize)
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class BlobModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):

        # Args:
        # input_features (int): Number of input features to the model
        # output_features (int): Number of outputs features (number of output classes)
        # hidden_units (int): Number of hidden units between layers, default 8

        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_4 = BlobModel(in_features=2, out_features=4, hidden_units=8).to(device)
# print(model_4)

# print(X_blob_train.shape, y_blob_train[:5])
# print(torch.unique(y_blob_train))

# Create a loss function for multi-class classification 
loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)

# Let's get some raw outputs of our model (logits)
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device)) # data in cpu! 

print(y_blob_test[:10])

# Convert our model's logit outputs to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)

print(y_logits[:5])
print(y_pred_probs[:5])

print(torch.sum(y_pred_probs[0]))
print(torch.argmax(y_pred_probs[0]))

with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device)) # data in cpu! 

# Convert our model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)

print(y_preds)
print(y_blob_test)

print(y_blob_train.dtype)

# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs 
epochs = 100

# Put data on the available device
X_blob_train, X_blob_test, y_blob_train, y_blob_test = map(lambda x: x.to(device), (X_blob_train, X_blob_test, y_blob_train, y_blob_test))

