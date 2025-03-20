import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Universal variable
imagesDir = './'

# Methods
def json_to_path(json_file):
    path_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)

        for row in data["images"]:
            p = row["path"]
            path_list.append(p)

    return path_list

def rotate_image(img):
    return

def transform_image(img):
    img = cv2.imread(os.path.join(imagesDir, p))
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gaussian_blur = cv2.GaussianBlur(gray_img,(5,5),0)
    plt.imshow(gaussian_blur,cmap="gray")

    plt.show()

def read_images(paths):
    for p in paths:
        transform_image(p)

# Application Logic
all_paths = json_to_path(os.path.join(imagesDir, 'test.json'))
read_images(all_paths)