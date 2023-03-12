import torch
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
import subprocess
import time

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
# More information on multiplying matrices https://www.mathsisfun.com/algebra/matrix—multiplying.html

# There are two main rules that performing "matrix mutliplication" needs to satisfy:

# 1. The inner dimensions must match:
#   • (3, 2) @ (3, 2) won't work
#   • (2, 3) @ (3, 2) will work
#   • (3, 2) @ (2, 3) will work

# 2. The resulting matrix has the shape of the outer dimensions:
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

start_time = time.time()
value = 0
for i in range(len(tensor)): 
    value += tensor [i] * tensor [i]
print (value)
end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")


start_time = time.time()
print(torch.matmul(tensor, tensor))
end_time = time.time() 
print("Execution time:", end_time - start_time, "seconds")


# one of the most common errors in deep learning (shape errors)