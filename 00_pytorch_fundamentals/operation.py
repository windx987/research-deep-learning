import torch
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
import subprocess
import time

# start_time = time.time()
# end_time = time.time()
# print("Execution time:", end_time - start_time, "seconds")

# manipulating tensors [tensor operations]
# Tensor opertions include: Addition, Subtraction, Multiplication, Division, Matrix multiplication

tensor = torch.tensor([1, 2, 3])

# Addition
print(tensor + 10)
print(torch.add(tensor, 10))

# Subtraction
print(tensor - 10)
print(torch.sub(tensor, 10))

# Multiplication (element-wise)
print(tensor * 10) # normal operations
print(torch.mul(tensor, 10)) # try out build-in functions

# Division
print(tensor / 10)
print(torch.div(tensor, 10))


# Two main ways of performing multiplication in neural networks and deep learning
# 1. Element-wise multilication
# 2. Matrix multiplication (dot.product)
# More information on multiplying matrices http://matrixmultiplication.xyz/

# There are two main rules that performing "matrix mutliplication" needs to satisfy:

# 1. The "inner dimensions" must match:
#   • (3, 2) @ (3, 2) won't work
#   • (2, 3) @ (3, 2) will work
#   • (3, 2) @ (2, 3) will work

# 2. The resulting matrix has the shape of the "outer dimensions":
#   • (2, 3) @ (3, 2) -> (2, 2)
#   • (3, 2) @ (2, 3) -> (3, 3)

#example
print("Example")
# torch.matmul(torch.rand(3, 2), torch.rand(2, 3) // won't work
print(torch.matmul(torch.rand(3, 2), torch.rand(2, 3)).shape) 

# Element-wise multilication
print(tensor, "*", tensor)
print(f"Equals: {tensor*tensor}")

# Matrix multiplication
print(torch.matmul(tensor, tensor)) 

value = 0
for i in range(len(tensor)): 
    value += tensor [i] * tensor [i]
print (value)

# one of the most common errors in deep learning (shape errors)

# shape for matrix multiplication

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

print(f"Shape of tensor: {tensor_A.shape}") # (3, 2)
print(f"Shape of tensor: {tensor_B.shape}") # (3, 2)

#print(torch.mm(tensor_A, tensor_B))    # same as torch.matmul() // (3, 2) @ (3, 2) won't work
print(torch.mm(tensor_A, tensor_B.T))   # (3, 2) @ (3, 2) will work with tensor.T

# To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a transpose.
# A transpose switches the axes or dimensions of a given tensor.
# By use torch.transpose(input, dim0, dim1) or tensor.T //dim (dimension)

print(tensor_B.shape)
print(tensor_B.T.shape)

print("\n") # http://matrixmultiplication.xyz/
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

# find min, max, mean, sum, etc (tensor aggregation)
x = torch.arange(0, 100, 10) # x.dtype is long
print(x)

# find min
print(f"Minimum: {x.min(), torch.min(x)}")

# find max
print(f"Maximum: {x.max(), torch.max(x)}")

# find mean
print(f"Mean: {torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()}")
# won't work without float or complex datatype because x.dtype is long

# RuntimeError: mean(): could not infer output dtype. 
# Input dtype must be either a floating point or complex dtype. Got: Long

# find sum
print(f"Sum: {x.sum(), torch.sum(x)}")

# find positional min/max #data is position of x
print(f"Index where min value occurs: {x.argmin()} value is", x[x.argmin()])
print(f"Index where max value occurs: {x.argmax()} value is", x[x.argmax()])


# Reshaping, stacking, squeezing and unsqueezing tensors
# • Reshaping - reshapes an input tensor to a defined shape
# • View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# • Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# • Squeeze - removes all 1 dimensions from a tensor
# • Unsqueeze - add a 1 dimension to a target tensor
# • Permute - Return a view of the input with dimensions permuted (swapped) in a certain way

x = torch.arange(1., 10., 1.)
print(x,x.shape)

# Add an extra dimension
x_reshaped = x.reshape(9, 1) # layer outer, layer inner
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(1, 9)  # layer outer, layer inner
print(x_reshaped, x_reshaped.shape)

# Change the view
z = x.view(1, 9)
print(z, z.shape)

# changing z changes x (because input tensor of certain shape but keep the same memory as the original tensor)
z[:, 0]= 5
print(z,x)
print("\n")

# stack tensors on top of each other
x_stracked = torch.stack([x, x, x, x], dim=0)
print(x_stracked, x_stracked.shape,"\n")
x_stracked = torch.stack([x, x, x, x], dim=1)
print(x_stracked, x_stracked.shape,"\n")



x_stracked = torch.hstack([x, x], out=None)
print(x_stracked, x_stracked.shape,"\n")
x_stracked = torch.vstack([x, x], out=None)
print(x_stracked, x_stracked.shape,"\n")

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped with squeeze
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}\n")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}\n")

# torch.permuted is rearanges the dimensions of a target tensor in a specified order
# Create tensor with specific shape // torch.permute(input, dims)
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}\n") # [colour_con]

x_original[0, 0, 0] = 728218
print(x_original[0, 0, 0], x_permuted[0, 0, 0])

# the colon ":" is used as a shorthand for "all indices" or "all dimensions"
print(x_original[0, 0, :])

# Indexing (selecting data from tensors)
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape,"\n")

# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}\n")


# Get all values of 0th dimension and the 0 index of 1st dimension
print(x[:, 0])

# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
print(x[:, :, 1])

# Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
print(x[:, 1, 1])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension 
print(x[0, 0, :]) # same as x[0][0]

#Exercises index
# Index on x to return 9, [9]
print(x[0, 2, 2], x[:, 2, 2])
# Index on x to return [3, 6, 9] , [[3, 6, 9]]
print(x[0, :, 2], x[:, :, 2])