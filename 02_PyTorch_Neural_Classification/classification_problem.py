import torch
from torch import nn
from sklearn.datasets import make_circles

model = nn.Sequential(
        nn.Linear(in_features=3, out_features=100,),
        nn.Linear(in_features=100, out_features=100,),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=3,),)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)