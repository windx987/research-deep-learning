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

fig = plt.figure(figsize=(3, 3))
row, cols = 3, 3
for i in range(1, row*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    img_squeeze = img.squeeze()
    fig.add_subplot(row, cols, i)
    plt.imshow(img_squeeze, cmap="gray")
    plt.title(label=class_names[label])
    plt.axis(False)

plt.show()

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Setup dataloader
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoder:{train_dataloader, test_dataloader}")
print(f"Lenght:{len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Lenght:{len(test_dataloader)} batches of {BATCH_SIZE}...")
print("\n")

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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(f"Output shape of conv block 1: {x.shape}")
        x = self.conv_block_2(x)
        print(f"Output shape of conv block 2: {x.shape}")
        x = self.conv_block_3(x)
        print(f"Output shape of conv block 3: {x.shape}")
        x = self.classifier(x)
        print(f"Output shape of classifier: {x.shape}")
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MNIST_model(input_shape=1,
                    hidden_units=10,
                    output_shape=10).to(device)

# Check out the model state dict to find out what patterns our model wants to learn
print(model)
