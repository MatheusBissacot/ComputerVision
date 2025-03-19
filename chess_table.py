import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def open_image(path):
    img = cv2.imread(os.path.join(imagesDir, path))

    # Resize image to facilitate visualization
    img = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)

    # Show image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Application Logic
all_paths = json_to_path(os.path.join(imagesDir, 'test.json'))
for path in all_paths:
    open_image(path)