from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input + 1
    

Mymodel = Model()

x = torch.tensor(1)

output = Mymodel(x)

print(output)