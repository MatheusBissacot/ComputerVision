import cv2
import numpy as np
import json
import os

# Directories
imagesDir = './VC_2425_Project_public/'
outputDir = './output_chessboard/'
os.makedirs(outputDir, exist_ok=True)


# Load image paths from JSON
def json_to_path(json_file):
    path_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for row in data["images"]:
            path_list.append(row["path"])
    return path_list


# Find the chessboard table in the image
def detect_chess_table(img_path):
    img = cv2.imread(os.path.join(imagesDir, img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Detect edges
    edges = cv2.Canny(thresh, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    chessboard_contour = None

    # Find the largest quadrilateral
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Ensuring it's a quadrilateral
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                chessboard_contour = approx

    if chessboard_contour is not None:
        return img, chessboard_contour
    else:
        return img, None


# Crop the chessboard (No Warping)
def crop_chess_table(img_path):
    img, chessboard_contour = detect_chess_table(img_path)

    if chessboard_contour is None:
        print(f"No chessboard detected in {img_path}")
        return None

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(chessboard_contour)

    # Crop image
    chessboard_cropped = img[y:y+h, x:x+w]

    # Save the cropped chessboard
    output_path = os.path.join(outputDir, f"{os.path.basename(img_path)}_cropped.jpg")
    cv2.imwrite(output_path, chessboard_cropped)
    print(f"Saved cropped chessboard: {output_path}")

    return chessboard_cropped


# Hough Transform for Line Detection
def find_chessboard_lines(image_path):
    print("Processing image:", image_path)
    image = cv2.imread(os.path.join(imagesDir, image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel and Canny
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    _, binary = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(np.uint8(binary), 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    line_img = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save image with detected lines
    cv2.imwrite(os.path.join(outputDir, f"{os.path.basename(image_path)}_lines.jpg"), line_img)
    
    return line_img, lines

# Process images
def process_chessboards(json_file):
    paths = json_to_path(json_file)
    for p in paths:
        # crop_chess_table(p)
        find_chessboard_lines(p)

# Run pipeline
test_json_path = os.path.join('./test.json')
process_chessboards(test_json_path)