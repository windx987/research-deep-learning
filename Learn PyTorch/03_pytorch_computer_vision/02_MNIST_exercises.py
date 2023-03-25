    # 1. What are 3 areas in industry where computer vision is currently being used?
# answer >> 1. self driving car 
#           2. pick-place product 
#           3. authorization face recognition

# 2. Search "what is overfitting in machine learning" and write down a sentence about what you find.
# answer >> Overfitting is an undesirable machine learning behavior that occurs 
#           when the machine learning model gives accurate predictions for training data but not for new data

# 3. Search "ways to prevent overfitting in machine learning", write down 3 of the things you find and a sentence about each.
# answer >> You can prevent overfitting by diversifying and scaling your training data set or using some other data science strategies, like those given below.
            # 1. Early stopping
            # 2. Pruning
            # 3. Regularization
            # 4. Ensembling
            # 5. Data augmentation

# 4. Spend 20-minutes reading and clicking through the CNN Explainer website.

import torch
from torch import nn

# Import helper functions
from helper_function import accuracy_fn

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import helper functions
from helper_function import accuracy_fn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

# Import matplotlib for visualization
import numpy as numpy
import matplotlib.pyplot as plt

from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup training data 
train_data = datasets.MNIST(root="data", train=True, transform=ToTensor(), target_transform=None, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=ToTensor(), target_transform=None, download=True)
class_names = train_data.classes 
class_to_idx = train_data.class_to_idx 

# see the first training dataset
image, label = train_data[0]

# Check the shape of our image
print(f"Image shape: {image.shape} -> [color_channels, height, width]") 
print(f"Image label: {class_names[label]}")

# fig = plt.figure(figsize=(3, 3))
# row, cols = 3, 3
# for i in range(1, row*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     img_squeeze = img.squeeze()
#     fig.add_subplot(row, cols, i)
#     plt.imshow(img_squeeze, cmap="gray")
#     plt.title(label=class_names[label])
#     plt.axis(False)

# plt.show()

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Setup dataloader
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoder:{train_dataloader, test_dataloader}")
print(f"Lenght:{len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Lenght:{len(test_dataloader)} batches of {BATCH_SIZE}...")
print("\n")

def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
  train_loss, train_acc = 0, 0

  model.train()

  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss # accumulate train loss
    train_acc += accuracy_fn(y_true=y,
                             y_pred=y_pred.argmax(dim=1)) # go from logits -> prediction labels
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 400 == 0:
      print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
  
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}")

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

      test_pred = model(X)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y,
                              y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels 

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}\n")

# create CCN
class MNIST_model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # multiple with flatten of the above layer
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

device = "cpu"
print(f"Using device: {device}")

model_cpu = MNIST_model(input_shape=1,
                    hidden_units=10,
                    output_shape=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_cpu.parameters(), lr=0.1)

# Measure time
train_time_start_on_cpu = timer() 

epochs = 0
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model=model_cpu, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(model=model_cpu, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
    
# Measure time
train_time_end_on_cpu = timer()
total_train_time_on_cpu = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=device)


# train on gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_gpu = MNIST_model(input_shape=1,
                    hidden_units=10,
                    output_shape=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_gpu.parameters(), lr=0.1)

# Measure time
train_time_start_on_gpu = timer() 

epochs = 0
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model=model_gpu, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(model=model_gpu, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
    
# Measure time
train_time_end_on_gpu = timer()
total_train_time_on_gpu = print_train_time(start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=device)

import random

test_samples = [] 
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    # Get image and labels from the test data
    img, label = test_data[i]

    # Plot the sample
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")

    # Make prediction on image
    model_pred_logits = model_gpu(img.unsqueeze(dim=0).to(device))
    model_pred_probs = torch.softmax(model_pred_logits, dim=1)
    model_pred_label = torch.argmax(model_pred_probs, dim=1)

    # Plot the prediction
    title_text = plt.title(f"Truth: {label} | Pred: {model_pred_label.cpu().item()}")

    # Check for equality between pred and truth and change color of title text
    if label == model_pred_label:
        plt.title(title_text, fontsize=10, c="g") 
    else:
        plt.title(title_text, fontsize=10, c="r") 
    plt.axis(False)

plt.show()

y_preds = []
model_gpu.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions..."):
    # Send the data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_gpu(X)
    # Turn predictions from logits -> prediction probabilities -> prediction labels
    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
    # Put prediction on CPU for evaluation
    y_preds.append(y_pred.cpu())

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=torch.cat(y_preds),
                         target=test_data.targets)

plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10, 7))
# plt.show()

## various hyperparameter settings ##
random_tensor = torch.rand([1, 3, 64, 64])

conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=64,
                       kernel_size=3,
                       stride=2,
                       padding=1)

print(f"Random tensor original shape: {random_tensor.shape}")
random_tensor_through_conv_layer = conv_layer(random_tensor)
print(f"Random tensor through conv layer shape: {random_tensor_through_conv_layer.shape}")