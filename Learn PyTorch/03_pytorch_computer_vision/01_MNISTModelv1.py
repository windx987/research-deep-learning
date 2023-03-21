# Import PyTorch
import torch
from torch import nn

# Import helper functions
from helper_function import accuracy_fn

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

def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

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

print(train_data, test_data)
print(len(train_data), len(test_data))

image, label = train_data[0]
# print(image, label)
class_names = train_data.classes 
# print(class_names)
class_to_idx = train_data.class_to_idx 
# print(class_to_idx)

print(f"Image shape: {image.shape} -> [color_channels, height, width]") 
print(f"Image label: {class_names[label]}")

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

print(f"DataLoder:{train_dataloader, test_dataloader}")
print(f"Lenght:{len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Lenght:{len(test_dataloader)} batches of {BATCH_SIZE}...")
print("\n")

flatten_model = nn.Flatten()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape:int,
                  hidden_units:int, 
                  output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, hidden_units=10, output_shape=len(class_names)).to(device) 
next(model_1.parameters()).device

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
    
        loss = loss_fn(y_pred, y) # (per batch)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                             y_pred=y_pred.argmax(dim=1)) # go from logits -> prediction labels
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # (update the model's parameters once *per batch*)
  
    train_loss /= len(data_loader)
    train_acc  /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X) # forward pass

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, 
                                    y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels 
    test_loss /= len(data_loader)
    test_acc  /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

torch.manual_seed(42)
train_time_start_on_gpu = timer() 

epochs = 3

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n----------")
  train_step(model=model_1,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model_1,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)