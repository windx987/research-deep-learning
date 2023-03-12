import torch
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt

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
random_image_size_tensor = torch.rand(size=(224, 224, 3)) #defult is size=
print(random_image_size_tensor.shape,random_image_size_tensor.ndim)
print(random_image_size_tensor[0][1])

#zero and one
zeros = torch.zeros(3, 4)
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

#tensor datatypes

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,
                               device=None,
                               requires_grad=False)
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)