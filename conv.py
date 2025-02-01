import torch.nn.functional as F
import torch


input = torch.tensor([[1,1,1,1],
                      [1,1,1,1],
                      [1,1,1,1],
                      [1,1,1,1]]).reshape((1,1,4,4))

kernel = torch.tensor([[1,1],
                       [1,1],]).reshape((1,1,2,2))

print(input.shape)

output = F.conv2d(input, kernel, stride=1, padding=1)

print(output)