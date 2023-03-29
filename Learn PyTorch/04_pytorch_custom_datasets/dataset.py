import torch 
from torch import nn

# Note: PyTorch 1.10.0+ is required for this course
print(torch.__version__)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

import requests
import zipfile
from pathlib import Path

# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# # If the image folder doesn't exist, download it and prepare it...
# if image_path.is_dir():
#   print(f"{image_path} directory already exists... skipping download")
# else: 
#   print(f"{image_path} does not exist, creating one...")
#   image_path.mkdir(parents=True, exist_ok=True)

# # Download pizza, steak and suhsi data
# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#   request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#   print("Downloading pizza, steak, suhsi data...")
#   f.write(request.content)

# # Unzip pizza, steak, sushi data
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#   print("Unzipping pizza, steak and sushi data...")
#   zip_ref.extractall(image_path)

import os
def walk_through_dir(dir_path):
  """Walks through dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

print(train_dir, test_dir)

import random 
from PIL import Image

# Set seed
# random.seed(42)

# 1. Get all image paths 
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Pick a random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata 
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img.show()