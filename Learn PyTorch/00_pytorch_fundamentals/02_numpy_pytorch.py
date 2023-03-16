import torch
import numpy as np

array = np.array([1.0, 8.0])
tensor = torch.from_numpy(array) # numpy's default datatype of float64
# warning: when converting from numpy —> pytorch
# pytorch reflects numpy's default datatype of float64 unless specified otherwise
print(array, tensor)
print(array.dtype, tensor.dtype)

tensor = torch.from_numpy(array).type(torch.float32) # set datatype to float32
print(array, tensor)
print(array.dtype, tensor.dtype)

# Change the array, keep the tensor
array = array + 1
print(array, tensor)

# tensor to NumPy array
tensor = torch.ones(7) # pytorch's default datatype of float32
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)
print(tensor.dtype, numpy_tensor.dtype)

# Change the tensor, keep the array the same
tensor = tensor + 1
print(tensor, numpy_tensor)
print(tensor.dtype, numpy_tensor.dtype)

# Reproducibility (trying to take random out of random )

print(torch.rand(3, 3))

# In short how a neural network learns :
# start with random numbers -> tensor operations -> update random numbers to try and
# make them better representations of the data -> again -> again -> again...

# To reduce the randomness in neural networks and PyTorch comes the concept of a random seed.
# Essentially what the random seed does is "flavour" the randomness.

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere):\n{random_tensor_A == random_tensor_B}\n")

# Let's make some random but reproducible tensors

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere):\n{random_tensor_C == random_tensor_D}\n")

import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

# Check for GPU access with PyTorch
print(torch.cuda.is_available())

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Count number of devices
print(torch.cuda.device_count())

# For PyTorch since it's capable of running compute on the GPU or CPU, it's best practice to setup device agnostic code:
# https://pytorch.org/docs/stable/notes/cuda.html#best-practices

# putting tensors (and models) on the GPU
# The reason we want our tensors/models on the GPU is because using a GPU results in faster computations.


# create a tensor (default on the CPU)
tensor = torch.tensor([1, 2, 3])
 
# tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# Moving tensors back to the CPU
# NumPy use must use CPU not GPU

# if tensor is on gpu, can't transform it to NumPy
tensor_on_gpu.numpy() # won't work

# To fix the GPU tensor with NumPy issue, we can first set it to the CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
print(tensor_on_gpu)
