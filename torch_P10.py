import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


train_set = torchvision.datasets.CIFAR10(root='./torchdata', train = True,transform=dataset_trans)
test_set = torchvision.datasets.CIFAR10(root='./torchdata', train=False,transform=dataset_trans)

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass
# step = 0
# writer = SummaryWriter('dataloader')
# for data in test_loader:
#     imgs, targets = data
#     writer.add_images('test_loader', imgs, step)
#     step += 1

# writer.close()
# writer = SummaryWriter('P10')


# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image('test_set', img, i)

# writer.close()

# print(test_set[0])
# print(test_set.classes)
# test_set[0][0].show()
  