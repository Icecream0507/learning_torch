from PIL import Image
from torchvision import transforms
import torch
#from torch_P10 import Model

result_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_path = 'dog.jpg'
image = Image.open(image_path)

print(image.size)

image = image.crop((500, 1000, 2000, 2500)).resize((32, 32))
image.show()
print(image.size)


tensor_trans = transforms.ToTensor()


trans = transforms.Compose([tensor_trans])

input = trans(image).reshape((1, 3, 32, 32))

print(input.shape)


model = torch.load('model_30.pth')

print(model)


output = model(input)

target = output.argmax(1)

print(target)

print(output)
print(result_list[target])

