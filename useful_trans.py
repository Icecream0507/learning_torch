from torchvision import transforms
from PIL import Image


img_path = 'train/ants_image/5650366_e22b7e1065.jpg'

img = Image.open(img_path)

print(type(img))




tensor_trans = transforms.ToTensor()

img_tensor = tensor_trans(img)

print(img_tensor.shape)

  