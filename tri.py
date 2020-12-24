import numpy as np
import matplotlib.pyplot as plt
# !pip install cv2
import os
import cv2


DATADIR = r"D:\\Neeraj\\Programming\\Python\Deep Learning\\Projects\dataset"
Classes = ['with_mask', 'without_mask']
Class_dict = {'with_mask':1, 'without_mask':0}
img_size = 100

data = []
target = []

# DATADIR = "dataset/"
IMAGE_SIZE = 32
images_all_path = []
TARGET = [0,0,0,0,0]

# IMG_ARRAIES = []

# CATEGORIES = ["dogs", "cats"]

for category in Classes:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        image_path = os.path.join(path,img)
        img_array = cv2.imread(image_path)
        # print(os.path.join(path,img))
#         gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)           
        # resized=cv2.resize(img_array,(img_size,img_size))
        images_all_path.append(image_path)
#         data.append(resized)


