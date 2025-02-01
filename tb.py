from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


image_path = './hymenoptera_data/hymenoptera_data/train/bees/452462695_40a4e5b559.jpg'
img = Image.open(image_path)

img_array = np.array(img)

writer = SummaryWriter('logs')

writer.add_image('bee', img_array, 1, dataformats="HWC")

# for i in range(100):
#     writer.add_scalar('y = 3x', 3*i, i)

writer.close()