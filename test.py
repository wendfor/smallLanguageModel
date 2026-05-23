import numpy as np
import torch
def test(x):
  
  for j in x:
    def compute():
      return j#1,2,3
    j += compute()
    
  

x = torch.tensor([1,2,3])
print(test(x))
print(x)