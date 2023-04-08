
# 1. Documentation reading
# The documentation on torch.Tensor.
# The documentation on torch.cuda.

# 2. Create a random tensor with shape (7, 7).

# Import torch
import torch

# Create random tensor
tensor_A = torch.rand(7, 7)
print(tensor_A, tensor_A.shape)

# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).

# Create another random tensor
tensor_B = torch.rand(1, 7)

# Perform matrix multiplication 
tensor_C = torch.mm(tensor_A,tensor_B.T)
print(tensor_C, tensor_C.shape)

# 4. Set the random seed to 0 and do 2 & 3 over again.

# Set manual seed
torch.manual_seed(0)

# Create two random tensors
tensor_X = torch.rand(7, 7)
tensor_Y = torch.rand(1, 7)

# Matrix multiply tensors
tensor_Z = torch.mm(tensor_X,tensor_Y.T)
print(tensor_Z, tensor_Z.shape)

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? 
# (hint: you'll need to look into the documentation for torch.cuda for this one)
# If there is, set the GPU random seed to 1234.

# Set random seed on the GPU
torch.cuda.manual_seed(1234)

# Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). 
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).

# Set random seed
torch.cuda.manual_seed(1234)

# Check for access to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create two random tensors on GPU
tensor_A = torch.rand(2, 3).to(device)
tensor_B = torch.rand(2, 3).to(device)
print(tensor_A, "\n", tensor_B)

# 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).

# Perform matmul on tensor_A and tensor_B
tensor_Z = torch.mm(tensor_A,tensor_B.T)
print(tensor_Z, tensor_Z.shape)

# 8. Find the maximum and minimum values of the output of 7.

# Find max
print(f"max: {torch.max(tensor_Z)}")

# Find min
print(f"min: {torch.min(tensor_Z)}")

# 9. Find the maximum and minimum index values of the output of 7.

# Find arg max
print(f"arg max: {tensor_Z.argmax()}")

# Find arg min
print(f"arg min: {tensor_Z.argmin()}")

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). 
# Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

# Set seed
torch.cuda.manual_seed(7)

# Create random tensor
tensor_A = torch.rand(1, 1, 1, 10)

# Remove single dimensions
tensor_B = tensor_A.squeeze()

# Print out tensors and their shapes
print(tensor_A, tensor_A.shape)
print(tensor_B, tensor_B.shape)

tensor_C = tensor_B.unsqueeze(0) # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
print("\n", tensor_C, tensor_C.shape)