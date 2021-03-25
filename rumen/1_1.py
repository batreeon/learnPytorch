from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)

x = torch.rand(5,3)
print(x)

x = x.new_ones(5,3,dtype=torch.double)
print(x)

print(x.size())

# tuple
torch.Size([5,0])

y = x.view(15)
z = x.view(-1,5)
print(x.size(),y.size(),z.size())

b = z.numpy()
print(b)