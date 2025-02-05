import torchvision
import torch


vgg16 = torchvision.models.vgg16(weights=False)


torch.save(vgg16, 'vgg16_method1.pth')


torch.save(vgg16.state_dict(), 'vgg16_method2.pth')