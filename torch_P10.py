import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
import torch

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1 = MaxPool2d(kernel_size=2, ceil_mode=True)
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = MaxPool2d(kernel_size=2, ceil_mode=True)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = MaxPool2d(kernel_size=2, ceil_mode=True)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=10)
        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
            ,MaxPool2d(kernel_size=2, ceil_mode=True)
            ,Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
            ,MaxPool2d(kernel_size=2, ceil_mode=True)
            ,Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
            ,MaxPool2d(kernel_size=2, ceil_mode=True)
            ,nn.Flatten()
            ,nn.Linear(in_features=1024, out_features=64) 
            ,nn.Linear(in_features=64, out_features=10)
            ,nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


Mymodel = Model()

train_set = torchvision.datasets.CIFAR10(root='./torchdata', train = True,transform=dataset_trans)
test_set = torchvision.datasets.CIFAR10(root='./torchdata', train=False,transform=dataset_trans)

print(f'len(train_set) = {len(train_set)}')
print(f'len(test_set) = {len(test_set)}')


print(train_set.classes)


train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)



def train():
    learn_rate = 0.01
    optimizer = torch.optim.SGD(Mymodel.parameters(), lr=learn_rate)


    print(Mymodel)
    loss = nn.CrossEntropyLoss()


    total_train_step = 0 # 记录训练次数
    total_test_step = 0 # 记录测试次数
    epoch = 50 # 训练轮数

    writer = SummaryWriter('First_train')

    for i in range(epoch):
        print(f'--------第{i+1}轮训练开始--------')
        for data in train_loader:
            imgs, targets = data
            outputs = Mymodel(imgs)
            result_loss = loss(outputs, targets)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f'训练次数：{total_train_step}, loss = {result_loss.item()}')
                writer.add_scalar('train_loss', result_loss.item(), total_train_step)

        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                outputs = Mymodel(imgs)
                result_loss = loss(outputs, targets)
                total_test_loss += result_loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy
        print(f'整体测试集上的loss:{total_test_loss}')
        print(f'整体测试集上的正确率：{total_accuracy / len(test_set)}')
        writer.add_scalar('test_loss', total_test_loss, total_test_step)
        writer.add_scalar('test_accuracy', total_accuracy / len(test_set), total_test_step)
        total_test_step += 1

        if total_accuracy / len(test_set) > 8 or i == 30:
            torch.save(Mymodel, 'model_{}.pth'.format(i))
            print('模型已保存')
            break


    writer.close()

# train()


# input = torch.ones((64, 3, 32, 32))
# output = Mymodel(input)
# print(output.shape)

# writer = SummaryWriter('graph' )
# writer.add_graph(Mymodel, input)
# writer.close()

# step = 0
# writer = SummaryWriter('maxpooling')
# for data in test_loader:
#     imgs, targets = data
#     output = Mymodel(imgs)
#     # print(imgs.shape)
#     # print(output.shape)
#     writer.add_images('input', imgs, step)
#     writer.add_images('output', output, step)
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
  