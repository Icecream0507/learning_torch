import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -1],
                      [2, -7]])


input = input.reshape((-1, 1, 2, 2))


dataset = torchvision.datasets.CIFAR10('torchdata', download=True, 
                                       transform=torchvision.transforms.ToTensor(),
                                       train= True)


dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64)



class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.Sigmoid()
    
    def forward(self, input):
        return self.relu(input)


Mymodel = Model()

writer = SummaryWriter("nn_relu")

step = 0
for data in dataloader:
    imgs, targets = data
    output = Mymodel(imgs)
    writer.add_images('input', imgs, step)
    writer.add_images('output', output, step)
    step += 1

writer.close()