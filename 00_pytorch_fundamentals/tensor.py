import torch
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
import subprocess
import time


# result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
# print(result.stdout.decode('utf-8'))

print(torch.__version__)
print(torch.cuda.is_available())

#scalar
scalar  = torch.tensor(7)
print(scalar) 
print(scalar.ndim) # 0 dimensions
print(scalar.item()) # can be use only 0-1 dimensions

#vector
vector  = torch.tensor([7,7])
print(vector)
print(vector.ndim) # 1 dimensions

#matrix
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim) # 2 dimensions
print(MATRIX[0]) #info in array index 0
print(MATRIX.shape) #2x2

#tensor
TENSOR  = torch.tensor([[[1, 2, 3],
                         [3, 6, 9],
                         [2, 4, 5]]])
print(TENSOR.ndim) # 3 dimensions
print(TENSOR.shape) # 1,3,3
print(TENSOR[0])
print(TENSOR[0][1])
print(TENSOR[0][1][2])

#random tensor
random_tensor = torch.rand(3, 4) #torch.rand(4, 4, 4)
print(random_tensor)

#create random tensor of [height, width, color_channels]
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape,random_image_size_tensor.ndim)
print(random_image_size_tensor[0][1])

#zero and one
zeros = torch.zeros(3, 4) # by defult is size=
print(zeros, zeros.dtype)

ones = torch.ones(3, 4)
print(ones, ones.dtype)

print(ones*random_tensor)
print(zeros*random_tensor)

#create range and tensors-like
one_to_ten = torch.arange(0,10,1) # step=, end=, step= defult is 1
print(one_to_ten)

#create tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

#tensor datatypes more in link https://pytorch.org/docs/stable/tensors.html#data-types


float_32_tensor = torch.tensor([3.5, 6.5, 9.5],
                               dtype=None, # defult is float 16
                               device=None, # 2 type are cpu, cuda >> cuda:0 //defult is cpu.
                               requires_grad=False) # recorded tensor
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

float_16_tensor = float_32_tensor.type(torch.float16) 
print(float_16_tensor)

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor)

mul=float_32_tensor*int_32_tensor
print(mul, mul.dtype)

# Getting information from tensors
# 1. Tensors not right datatype - to do get datatype from a tensor, can use "tensor.dtype"
# 2. Tensors not right shape - to get shape from a tensor, can use "tensor.shape"
# 3. Tensors not on the right device - to get device from a tensor, can use "tensor.device"

some_tensor = torch.rand([3, 4], dtype=torch.double, device='cuda:0')
print(some_tensor)
print(some_tensor.size(),some_tensor.shape) # does the same thing but one is function and the one is attribute

# find out details about some tensor

print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")
