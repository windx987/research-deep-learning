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

# Addition
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

# Subtraction
print(tensor - 10)

# Multiplication (element-wise)
print(tensor * 10) # normal operations
print(torch.mul(tensor, 10)) # try out build-in functions

# Division
print(tensor / 10)


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
# By use torch.transpose(input, dim0, dim1) or tensor.T

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
print(x_stracked)
x_stracked = torch.stack([x, x, x, x], dim=1)
print(x_stracked)
print("\n")

x_stracked = torch.hstack([x, x], out=None)
print(x_stracked)
x_stracked = torch.vstack([x, x], out=None)
print(x_stracked)

#page 25 bye bye sleep well