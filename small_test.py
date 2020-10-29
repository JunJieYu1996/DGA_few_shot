import tensorflow as tf
import torch

print("aaa")
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[1],[5]])

c = a.resize_(1,2)

d=torch.sum(c,dim=0)

print("123")