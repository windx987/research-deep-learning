# NumPy array to tensor
import torch
import numpy as np

array = np.array(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)