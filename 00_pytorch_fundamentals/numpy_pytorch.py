import torch
import numpy as np

arr = np.array([1.0, 8.0])
tensor = torch.from_numpy(arr)
print(arr, tensor)