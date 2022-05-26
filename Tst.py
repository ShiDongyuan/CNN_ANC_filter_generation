import torch
import numpy as np

c = torch.tensor([3, 4, 5, 6],dtype=torch.float32)
print(len(c))

As = torch.tensor([[1,2,3],
                   [1,2,3],
                   [1,2,3]])
Bs = torch.tensor([[1,0,3],
                   [1,1,3],
                   [1,2,3]])
d  = torch.einsum('NC,NC->N',Bs,Bs)

print(1/d)

c = np.array([[1, 2],
             [3,4]])
d = np.array([[1],[2]])

print(d*c)
