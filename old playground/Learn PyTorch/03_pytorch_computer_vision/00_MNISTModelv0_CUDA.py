# Import PyTorch
import torch
from torch import nn

# Import helper functions
from helper_function import plot_predictions, plot_decision_boundary, accuracy_fn

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

# Import matplotlib for visualization
import matplotlib.pyplot as plt

from timeit import default_timer as timer

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# use dataset FASHIONMNIST

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=False, # do we want the train or test dataset?
    download=True, # do we want to download yes/no?
    transform=transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None # how do we want to transform the labels/targets?
)

# print data
print(train_data, test_data)
print(len(train_data), len(test_data))

# see the first training dataset
image, label = train_data[0]
# print(image, label)
class_names = train_data.classes 
# print(class_names)
class_to_idx = train_data.class_to_idx 
# print(class_to_idx)

# Check the shape of our image
print(f"Image shape: {image.shape} -> [color_channels, height, width]") 
print(f"Image label: {class_names[label]}")

# visualize the image
image, label = train_data[0]
print(f"\nImage shape: {image.shape}")
# plt.imshow(image.squeeze())
# plt.title(label)
# plt.show()

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# torch.manual_seed(42)

# plot more images
# fig = plt.figure(figsize=(9, 9))
# row, cols = 4, 4
# for i in range(1, row*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     print(random_idx)
#     img, label = train_data[random_idx]
#     fig.add_subplot(row, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()

# Setup the batch size hyperparameter
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

# Let's check out what what we've created
print(f"DataLoder:{train_dataloader, test_dataloader}")
print(f"Lenght:{len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Lenght:{len(test_dataloader)} batches of {BATCH_SIZE}...")
print("\n")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)

# Show a sample
torch.manual_seed(42)

# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)

# print(f"Image size: {img.shape}")
# print(f"Label: {label}, label size: {label.shape}")

# plt.show()

# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x  = train_features_batch[0]

# Flatten the sample
out = flatten_model(x)

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {out.shape} -> [color_channels, height*width]")
# print(out.squeeze().shape)


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_uints:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_uints),
            nn.Linear(in_features=hidden_uints, out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)

model_0 = FashionMNISTModelV0(
    input_shape=28*28, # this is 784
    hidden_uints=10, # how many uints in hidden layer
    output_shape=len(class_names) # how many uints in output layer
).to(device)

# print(model_0)

# dummy input
# dummy_x = torch.rand([1, 1, 28, 28])
# print(dummy_x.shape)

# print(model_0(dummy_x).shape)
# print(model_0.state_dict())

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # multi-class data, loss function will be nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), # (stochastic gradient descent)
                            lr=0.1)

def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

'''
start_time = timer()
# some code...
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")
'''

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer() 

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
### training
    train_loss = 0 
### Add a loop to loop through the training batches
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_0.train()
    ## 1. Forward pass
        y_pred = model_0(X)
    ## 2. Calcurate the loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
    ## 3. Optimizer zero grad
        optimizer.zero_grad()
    ## 4. Loss backward
        loss.backward()
    ## 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()
    ## Print out what's happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
### Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)
    print(f"Train loss: {train_loss:.3f}")
### testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
        ## 1. Forward pass
            test_pred = model_0(X_test)
        ## 2. Calcurate the loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)
        ## 3. Calculate accuracy
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
    ## Calculate the test loss average per batch
        test_loss /= len(test_dataloader)
    ## Calculate the test acc average per batch
        test_acc /= len(test_dataloader)
### Print out what's happening
print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


### Calculate training time
train_time_stop_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, end=train_time_stop_on_cpu, 
                                            device=str(next(model_0.parameters()).device))
print("\n")

# Make predictions and get Model results with functions

## testing function
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
        ## 1. Forward pass
            y_pred = model(X)
        ## 2. Calcurate the loss (accumulatively)
            loss += loss_fn(y_pred, y)
        ## 3. Calculate accuracy
            acc += accuracy_fn(y_true=y, 
                               y_pred=y_pred.argmax(dim=1))
    ### Scale loss and acc to find the average loss/acc per batch    
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

model_0_results = eval_model(model=model_0, 
                             data_loader=test_dataloader,
                             loss_fn=loss_fn, 
                             accuracy_fn=accuracy_fn)
print(model_0_results)