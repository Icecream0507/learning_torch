import torch
from PIL import Image


image_path = './hymenoptera_data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(image_path)
img.show()
print(img.size)




