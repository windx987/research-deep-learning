import torch
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

## Introduction of Tensors 

#scalar
scalar  = torch.tensor(7)
print(scalar)