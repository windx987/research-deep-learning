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

# Setup the batch size hyperparameter
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

# create CCN
class FashionMNISTModelV2(nn.Module):
  """Model """
  def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
    super().__init__()
    self.conv_block1 = nn.Sequential(
    nn.Conv2d(in_channels=input_shape, 
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1), # values we can set ourselves in our NN's are called hyperparameters
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block2 = nn.Sequential(
    nn.Conv2d(in_channels=hidden_units,
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=hidden_units,
              out_features=output_shape)
    )
  def forward(self, x):
    return self.classifier(self.conv_block2(self.conv_block1(x)))