import cv2
import numpy as np
import os
import json


imagesDir = './'
outputDir = './output_images/'  


if not os.path.exists(outputDir):
    os.makedirs(outputDir)

def json_to_path(json_file):
    path_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)

        for row in data["images"]:
            p = row["path"]
            path_list.append(p)

    return path_list

def transform_image(img, output_name):

    img = cv2.imread(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)

    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    

    sobel_edges = cv2.magnitude(sobel_x, sobel_y)


    sobel_edges = cv2.convertScaleAbs(sobel_edges)


    ret, otsu_binary = cv2.threshold(sobel_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    kernel = np.ones((7, 7), np.uint8) 
    img_dilation = cv2.dilate(otsu_binary, kernel, iterations=1) 

    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            cv2.line(img_dilation, (x1, y1), (x2, y2), (255, 255, 255), 2)

    kernel = np.ones((3, 3), np.uint8) 
    img_dilation_2 = cv2.dilate(img_dilation, kernel, iterations=1) 

    
    output_path = os.path.join(outputDir, output_name)
    cv2.imwrite(output_path, img_dilation_2)  
    print(f"Imagem salva em: {output_path}")

def read_images(paths):
    for i, p in enumerate(paths):
       
        output_name = f"output_image_{i + 1}.png"
        transform_image(p, output_name)


all_paths = json_to_path(os.path.join(imagesDir, 'image_paths.json'))
read_images(all_paths)
