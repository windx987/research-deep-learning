# Import PyTorch
import torch
from torch import nn

# Import helper functions
from helper_function import plot_predictions, plot_decision_boundary

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

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

train_dataloder = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
test_dataloder = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

# Let's check out what what we've created
print(f"DataLoder:{train_dataloder, test_dataloder}")
print(f"Lenght:{len(train_dataloder)} batches of {BATCH_SIZE}...")
print(f"Lenght:{len(test_dataloder)} batches of {BATCH_SIZE}...")
print("\n")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloder))
print(train_features_batch.shape, train_labels_batch.shape)

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
print(out.squeeze().shape())

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_uints:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape=input_shape, hidden_uints=hidden_uints),
            nn.Linear(in_features=hidden_uints, out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)
