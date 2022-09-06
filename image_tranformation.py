import os
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np


class ImgTransformation:
    def __init__(self, data_path, partition="val2017"):
        self.data_path = data_path
        self.dir = os.path.join(data_path, partition)
        self.file_list = os.listdir(self.dir)

    def read_file(self, file_name):
        """
        Given .txt file, read the file line by line
        return : list of str ('class coordinate')
        """
        f = open(os.path.join(self.dir, file_name), "r")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        f.close()

        return lines

    def color_change(self, file_name):
        src = cv2.imread(
            os.path.join(self.data_path, f"val2017/{file_name}"), cv2.IMREAD_UNCHANGED
        )
        red_img, green_img, blue_img = (
            np.zeros(src.shape),
            np.zeros(src.shape),
            np.zeros(src.shape),
        )

        if src.ndim >= 3:
            red_img[:, :, 2] = src[:, :, 2]
            green_img[:, :, 1] = src[:, :, 1]
            blue_img[:, :, 0] = src[:, :, 0]

            cv2.imwrite(f"/hdd/hdd3/coco_val_transform/red/{file_name}", red_img)
            cv2.imwrite(f"/hdd/hdd3/coco_val_transform/green/{file_name}", green_img)
            cv2.imwrite(f"/hdd/hdd3/coco_val_transform/blue/{file_name}", blue_img)
        else:
            print(f"array is {src.ndim}-dimensional")

    def hue_change(self, file_name, value):
        src = cv2.imread(
            os.path.join(self.data_path, f"val2017/{file_name}"), cv2.IMREAD_UNCHANGED
        )
        transforms.ColorJitter(hue=0.5)(src)

    def transfer_img(self, color=True, hue=False):
        for file in tqdm(self.file_list):
            if color:
                self.color_change(file)
            # if hue:


root = "/hdd/hdd3/coco_custom/images/"

trans = ImgTransformation(root, "val2017")

# src = Image.open(os.path.join(root, f'val2017/{trans.file_list[0]}'))
# src.save('test_non.jpg')
# # src = cv2.imread(os.path.join(root, f'val2017/{trans.file_list[0]}'), cv2.IMREAD_UNCHANGED)
# transforms.ColorJitter(brightness=1)(src)
# src.save('test.jpg')
# cv2.imwrite('test.jpg', src)
# print(trans.file_list)
trans.transfer_img()
