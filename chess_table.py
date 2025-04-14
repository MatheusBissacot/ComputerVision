'''
COMPUTER VISION - PROJECT 1

Authors: - Catarina Monteiro
- Diogo Mendes
- Gonçalo Brochado
- Matheus Bissacot

Objectives:
1. Detect the chessboard in the image
2. Create a JSON file with:
    1.- number of pieces
    2.- board matrix with 1s and 0s representing the piece/no piece respectively
    3.- positions of the pieces in the original image using bounding boxes

Input: JSON file with image paths (METER FUNÇÃO DE INPUT)
Output: JSON file with the results -> json_output()

Overview:
- json_to_path(json_file) - Reads a JSON file and extracts image paths
- read_images(paths) - Reads images from the paths
- transform_image(img) - Processes the image to detect edges and lines
- find_board_contour(binary_image) - Finds the largest quadrilateral contour in the image
- order_points(pts) - Orders points in a consistent manner
- validate_squares(black_image, image) - Validates and identifies squares on the chessboard
- transform_perspective(rect) - Applies a perspective transformation to the detected chessboard
- perspective_to_original(H, positions) - Transforms positions from perspective to original space
- detect_pieces(warped, positions, margin) - Detects pieces on the chessboard
- json_output(input_image, number_pieces, board, detected_pieces) - Creates a JSON output file with results
- main(input_file) - Main function to execute the script
'''


# Imports
import cv2
import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
import math


# Directory variables
outputDir = './output_images/' 

# Create output directory
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Methods

"""
Description:
    Reads a JSON file containing image paths and extracts the paths into a list.

Parameters:
    json_file (str): The path to the JSON file containing the image paths.

Returns:
    list: A list of image paths extracted from the JSON file.
"""
def json_to_path(json_file):
    path_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)

        for path in data["image_files"]:
            path_list.append(path)

    return path_list

"""
Description:
    Reads a list of image paths, processes each image, and saves the transformed output.

Parameters:
    paths (list): A list of image file paths to be processed.

Returns:
    None
"""
def read_images(paths):
    for i, p in enumerate(paths):
        output_name = f"output_image_{i + 1}.png"
        transform_image(p, output_name)

"""
Description:
    Processes an input image to detect edges, apply thresholding, and extract lines using Hough Transform.

Parameters:
    img (str): The path to the input image.

Returns:
    tuple: A tuple containing the processed binary image with detected lines and the original image.
"""
def transform_image(img):
    try:
        img = cv2.imread(img)
    except:
        img=img
    
    # Grayscale conversion
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Gaussian blur
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Sobel edge detection
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)

    # Thresholding
    ret, otsu_binary = cv2.threshold(sobel_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel_clean = np.ones((3, 3), np.uint8)
    otsu_clean = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel_clean)

    # Canny edge detection
    canny = cv2.Canny(otsu_clean,100,255)

    # Dilation
    kernel = np.ones((3, 3), np.uint8) 
    img_dilation = cv2.dilate(canny, kernel, iterations=1) 

    # Hough Lines
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=50)

    # Create an image that contains only black pixels
    black_image = np.zeros_like(img_dilation)

    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw only lines to the "black_image"
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=3)

    return black_image, img, otsu_binary

"""
Description:
    Finds the largest quadrilateral contour in a binary image that likely represents the chessboard.

Parameters:
    binary_image (numpy.ndarray): A binary image where the chessboard is expected to be detected.

Returns:
    numpy.ndarray or None: A 4x2 array of ordered points representing the corners of the detected chessboard 
                           (top-left, top-right, bottom-right, bottom-left). Returns None if no valid contour is found.
"""
def find_board_contour(binary_image, otsu_binary):
    # Find countours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, (255), 2)  # Desenha os contornos em branco
    kernel = np.ones((8,8), np.uint8)
    closed = cv2.morphologyEx(contour_image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, (255), 2) 

    max_area = 0
    best_cnt = None
    points = 3
    while max_area <= 50000:
        points += 1
        if points > 8:
            points = 4
            # If can't find the board, it makes the image even noisier 
            kernel_clean = np.ones((4, 4), np.uint8)
            otsu_clean = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel_clean)
            canny = cv2.Canny(otsu_clean,100,255)
            kernel = np.ones((3, 3), np.uint8) 
            img_dilation = cv2.dilate(canny, kernel, iterations=1) 
            lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=50)
            black_image2 = np.zeros_like(img_dilation)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(black_image2, (x1, y1), (x2, y2), (255, 255, 255), 2)
            kernel = np.ones((3, 3), np.uint8) 
            black_image2 = cv2.dilate(black_image2, kernel, iterations=3)
            contours, _ = cv2.findContours(black_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000:
                continue

            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == points and area > max_area and not any(0 in pts for pts in approx):
                max_area = area
                best_cnt = approx
    
    if len(best_cnt)!=4:
        best_cnt = cv2.convexHull(best_cnt)

    if best_cnt is not None:
        pts = best_cnt.reshape(4, 2)
        rect = order_points(pts)
        return rect
    else:
        return None

"""
Description:
    Orders a set of four points in a consistent order: top-left, top-right, bottom-right, and bottom-left.

Parameters:
    pts (numpy.ndarray): A 4x2 array of points representing the corners of a quadrilateral.

Returns:
    numpy.ndarray: A 4x2 array of ordered points in the format: top-left, top-right, bottom-right, bottom-left.
"""
def order_points(pts):
    # Orders points in the format: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_pts = pts[np.argsort(angles)]
    sorted_pts = sorted_pts.astype("float32")
    return sorted_pts


"""
Description:
    Validates and identifies squares on the chessboard by analyzing contours and filtering them based on geometric properties.

Parameters:
    black_image (numpy.ndarray): A binary image containing potential square contours.
    image (numpy.ndarray): The original image for visualization purposes.

Returns:
    list: A sorted list of valid square positions, where each position contains the center coordinates and the four corner points.
"""
def validate_squares(black_image, image):
    # Look for valid squares and check if squares are inside of board
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel)
    # find contours
    board_contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # blank image for displaying all contours
    all_contours_image= np.zeros_like(black_image)

    # Copy blank image for displaying all squares 
    squares_image = np.copy(image) 

    # blank image for displaying valid contours (squares)
    valid_squares_image = np.zeros_like(black_image)

    valid_squares_positions = []

    # loop through contours and filter them by deciding if they are potential squares
    for contour in board_contours:
        if 2000 < cv2.contourArea(contour) < 20000:

            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # if polygon has 4 vertices
            if len(approx) == 4:

                # 4 points of polygon
                pts = [pt[0].tolist() for pt in approx]

                # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

                #  Y values
                if index_sorted[0][1]< index_sorted[1][1]:
                    cur=index_sorted[0]
                    index_sorted[0] =  index_sorted[1]
                    index_sorted[1] = cur

                if index_sorted[2][1]> index_sorted[3][1]:
                    cur=index_sorted[2]
                    index_sorted[2] =  index_sorted[3]
                    index_sorted[3] = cur

                # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                pt1=index_sorted[0]
                pt2=index_sorted[1]
                pt3=index_sorted[2]
                pt4=index_sorted[3]

                # find rectangle that fits 4 point 
                x, y, w, h = cv2.boundingRect(contour)
                # find center of rectangle 
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                # calculate length of 4 side of rectangle
                l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)
    
    
                # Create a list of lengths
                lengths = [l1, l2, l3, l4]
                
                # Get the maximum and minimum lengths
                max_length = max(lengths)
                min_length = min(lengths)

                # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,  
                if (max_length - min_length) <= 35: # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                    valid_square=True
                else:
                    valid_square=False
    
                if valid_square:

                    # Draw the lines between the points
                    cv2.line(squares_image, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(squares_image, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(squares_image, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(squares_image, pt1, pt4, (255, 255, 0), 7)

                    # Draw only valid squares to "valid_squares_image"
                    cv2.line(valid_squares_image, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt1, pt4, (255, 255, 0), 7)

                    valid_squares_positions.append([center_x,center_y,pt1, pt2, pt3, pt4])
                
                # Draw only valid squares to "valid_squares_image"
                cv2.line(all_contours_image, pt1, pt2, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt2, pt3, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt3, pt4, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt1, pt4, (255, 255, 0), 7)
    
    sorted_coordinates = sorted(valid_squares_positions, key=lambda x: x[1], reverse=True)

    groups = []
    current_group = [sorted_coordinates[0]]

    for coord in sorted_coordinates[1:]:
        if abs(coord[1] - current_group[-1][1]) < 50:
            current_group.append(coord)
        else:
            groups.append(current_group)
            current_group = [coord]

    # Append the last group
    groups.append(current_group)

    # Step 2: Sort each group by the second index (column values)
    for group in groups:
        group.sort(key=lambda x: x[0])

    # Step 3: Combine the groups back together
    sorted_coordinates = [coord for group in groups for coord in group]

    sorted_coordinates[:10]

    for num in range(len(sorted_coordinates)-1):
        if abs(sorted_coordinates[num][1] - sorted_coordinates[num+1][1])< 50 :
            if sorted_coordinates[num+1][0] - sorted_coordinates[num][0] > 150:
                x=(sorted_coordinates[num+1][0] + sorted_coordinates[num][0])/2
                y=(sorted_coordinates[num+1][1] + sorted_coordinates[num][1])/2
                p1=sorted_coordinates[num+1][5]
                p2=sorted_coordinates[num+1][4]
                p3=sorted_coordinates[num][3]
                p4=sorted_coordinates[num][2]
                sorted_coordinates.insert(num+1,[x,y,p1,p2,p3,p4])

    return sorted_coordinates

"""
Description:
    Applies a perspective transformation to warp the detected chessboard region into a top-down view.

Parameters:
    rect (numpy.ndarray): A 4x2 array of ordered points representing the corners of the detected chessboard 
                          (top-left, top-right, bottom-right, bottom-left).

Returns:
    tuple: A tuple containing the warped image (numpy.ndarray) and the homography matrix (numpy.ndarray).
"""
def transform_perspective(rect, image):
    if rect is not None:  
        width, height = 800, 800  # Final size of the board
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        H = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, H, (width, height))
    return warped, H


"""
Description:
    Transforms a list of positions from a perspective-transformed space back to the original space using a given homography matrix.

Parameters:
    H (numpy.ndarray): A 3x3 homography matrix used for the transformation.
    positions (list of lists): A list of positions, where each position contains the four corner points of a square 
                                in the perspective-transformed space.

Returns:
    list: A list of transformed positions in the original space, where each position is represented as a list of 
          transformed corner points.
"""
def perspective_to_original(H, positions):
    H_inv = np.linalg.inv(H)
    original_positions = []

    for pos in positions:
        if pos is not None:
            pos = np.array(pos, dtype=np.float32)
            pos = (pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3])
            pos = pos + np.array([50, 50, 50, 50])
            x_min, y_min, x_max, y_max = pos
            points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.float32).reshape(1, -1, 2)
            transformed_points = cv2.perspectiveTransform(points, H_inv)
            xs = transformed_points[0][:, 0]
            ys = transformed_points[0][:, 1]
            new_box = (min(xs), min(ys), max(xs), max(ys))
            original_positions.append(new_box)

    return original_positions

"""
Description:
    Detects white and black regions/pieces in an image based on color thresholds and returns the bounding boxes
    around those regions. It applies a margin crop, replaces colors for visualization, and draws contours 
    and bounding boxes on the image.

Parameters:
    img (numpy.ndarray): The input image in BGR format to be processed.

Returns:
    list: A list of bounding boxes (x, y, w, h) for detected regions where:
        - x, y represent the top-left corner of the bounding box.
        - w, h represent the width and height of the bounding box.
"""
def all_bounding_boxes_2(img, warped):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    margin = 50
    h, w = warped.shape[:2]

    img = warped[margin:h - margin, margin:w - margin]

    white_lower = np.array([40, 70, 100])
    white_upper = np.array([150, 190, 210])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([30, 30, 30])

    white_mask = cv2.inRange(img, white_lower, white_upper)
    black_mask = cv2.inRange(img, black_lower, black_upper)

    img[white_mask > 0] = [255, 0, 0]
    img[black_mask > 0] = [0, 0, 255]

    masks = [white_mask, black_mask]
    bounding_boxes = []

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(img)
        cv2.drawContours(contour_image, contours, -1, (255), 2)

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh > 2000:
                bounding_boxes.append((x, y, bw, bh))
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    return bounding_boxes

"""
Description:
    Detects pieces on a chessboard by analyzing the grid cells of a perspective-transformed image.

Parameters:
    warped (numpy.ndarray): The perspective-transformed image of the chessboard.
    positions (list, optional): A list of positions for additional processing (not used in this function).
    margin (int, optional): The margin to remove from the borders of the image. Default is 50.

Returns:
    tuple: A tuple containing:
        - board_matrix (numpy.ndarray): An 8x8 matrix with 1s representing the presence of a piece and 0s otherwise.
        - pieces (int): The total number of detected pieces on the chessboard.
"""
def detect_pieces(warped, positions=None, margin=50):
    # Remove borders
    h, w = warped.shape[:2]
    roi = warped[margin:h - margin, margin:w - margin]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Binary threshold
    _, thresh = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

    # Morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Dilation to "bold" the piece areas
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.dilate(morph, dilate_kernel, iterations=1)

    # Dimensions of board
    height, width = morph.shape
    cell_h, cell_w = height // 8, width // 8

    board_matrix = np.zeros((8, 8), dtype=int)
    inner_scale = 0.5
    offset_h = int((1 - inner_scale) / 2 * cell_h)
    offset_w = int((1 - inner_scale) / 2 * cell_w)
    pieces = 0
    boxes_chess = []

    # Convert ROI to RGB for visualization
    roi_rgb = roi.copy()

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            cy1 = y1 + offset_h
            cy2 = y2 - offset_h
            cx1 = x1 + offset_w
            cx2 = x2 - offset_w

            # Center point (relative to the ROI)
            center_x = (cx1 + cx2) / 2
            center_y = (cy1 + cy2) / 2
            boxes_chess.append((center_x, center_y))

            # Draw grid cell being analyzed
            cv2.rectangle(roi_rgb, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

            central_cell = morph[cy1:cy2, cx1:cx2]
            white_pixels = cv2.countNonZero(central_cell)

            # Add white pixel count as text
            text_x = x1 + 5
            text_y = y1 + 20
            cv2.putText(roi_rgb, str(white_pixels), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if white_pixels > 880:
                board_matrix[row, col] = 1
                pieces += 1

    return board_matrix, pieces, boxes_chess

"""
Description:
    Identifies the darkest corner of an image by analyzing the average intensity of its four corners.
Parameters:
    image (numpy.ndarray): The input image in BGR format.
    margin (int, optional): The size of the square region to consider for each corner. Defaults to 100.
Returns:
    str: The name of the darkest corner ('top_left', 'top_right', 'bottom_right', or 'bottom_left').
"""
def find_darkest_corner(image, margin=100):
    h, w = image.shape[:2]

    # Extract the 4 corners of the image
    corners = {
        'top_left': image[0:margin, 0:margin],
        'top_right': image[0:margin, w-margin:w],
        'bottom_right': image[h-margin:h, w-margin:w],
        'bottom_left': image[h-margin:h, 0:margin]
    }

    # Calculate the average intensity in grayscale
    averages = {
        name: np.mean(cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY))
        for name, corner in corners.items()
    }

    # Return the darkest corner
    return min(averages, key=averages.get)

"""
Description:
    Rotates an image so that the specified corner becomes the bottom-left corner.
Parameters:
    image (numpy.ndarray): The input image to be rotated.
    origin_corner (str): The current position of the corner to be moved to the bottom-left. 
                            Accepted values are 'top_left', 'top_right', 'bottom_right', or 'bottom_left'.
Returns:
    numpy.ndarray: The rotated image with the specified corner moved to the bottom-left.
"""
def rotate_to_bottom_left(image, matrix, origin_corner):
    if origin_corner == 'top_left':
        # 90° counterclockwise
        return [list(row) for row in zip(*matrix)][::-1], cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif origin_corner == 'top_right':
        # 180°
        return [row[::-1] for row in matrix[::-1]], cv2.rotate(image, cv2.ROTATE_180)
    elif origin_corner == 'bottom_right':
        # 90° clockwise
        return [list(row)[::-1] for row in zip(*matrix)], cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Already in the bottom left corner
        return matrix, image
    
"""
Description:
    Finds the largest bounding box (by area) from a list that contains a given point.

Parameters:
    point (tuple): A tuple (px, py) representing the point to check.
    bounding_boxes (list): A list of bounding boxes (x, y, w, h).

Returns:
    tuple or None: The bounding box (x, y, w, h) that contains the point and has the largest area.
                    Returns None if no bounding box contains the point.
"""
def find_largest_bounding_box_containing_point(point, bounding_boxes):
    px, py = point
    largest_area = 0
    best_box = None
    for x, y, w, h in bounding_boxes:
        if x <= px <= x + w and y <= py <= y + h:
            area = w * h
            if area > largest_area:
                largest_area = area
                best_box = (x, y, w, h)
    return best_box

"""
Description:
    Associates detected pieces on a board matrix with the best matching bounding boxes based on 
    proximity to known square centers. Each piece's location is matched to the largest bounding box 
    that contains its center point.

Parameters:
    board (numpy.ndarray): An 8x8 matrix with 1s indicating the presence of pieces and 0s otherwise.
    all_boxes (list): A list of candidate bounding boxes (x, y, w, h).
    boxes_chess (list): A list of center points (x, y) for each square on the board, ordered row-wise.

Returns:
    list: A list of selected bounding boxes (x, y, w, h) corresponding to detected pieces.
            Boxes are removed from the input list as they are matched.
"""
def detected_boxes(board, all_boxes, boxes_chess):
    best_box = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                best_box.append(find_largest_bounding_box_containing_point(boxes_chess[i*8+j], all_boxes))
                if best_box[-1] != None:
                    all_boxes.remove(best_box[-1])
    return best_box

"""
Converts bounding boxes from (xmin, ymin, xmax, ymax) tuples to dictionary format.

Parameters:
    boxes (list): A list of bounding boxes in (xmin, ymin, xmax, ymax) format.

Returns:
    list: A list of dictionaries with keys 'xmin', 'ymin', 'xmax', 'ymax'.
"""
def convert_to_dict_format(boxes):
    return [
        {"xmin": float(x1), "ymin": float(y1), "xmax": float(x2), "ymax": float(y2)}
        for (x1, y1, x2, y2) in boxes
    ]

"""
Description:
    Creates a JSON output file containing the results of the chessboard analysis, including the number of pieces, 
    the board matrix, and the positions of detected pieces.

Parameters:
    input_image (str): The path to the input image.
    number_pieces (int): The total number of detected pieces on the chessboard.
    board (list of lists): An 8x8 matrix with 1s representing the presence of a piece and 0s otherwise.
    detected_pieces (list): A list of positions of detected pieces in the original image, represented as bounding boxes.

Returns:
    None
"""
def json_output(input_image, number_pieces, board, detected_pieces):
    new_data = {
        "image": input_image,
        "num_pieces": number_pieces,
        "board": board,
        "detected_pieces": detected_pieces
    }

    file_path = 'output.json'
    
    # Load existing content if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new data
    data.append(new_data)

    # Write everything back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


# Main logic
def main(input_file):
    try:
        all_paths = json_to_path(input_file)

        for path in all_paths:

            # Transform image to detect the board
            black_image, image, otsu_binary = transform_image(path)
            # Find the board contour 
            rect = find_board_contour(black_image, otsu_binary)
            # Applies a perspective transformation to warp the detected chessboard region into a top-down view. 
            warped, H = transform_perspective(rect, image)
            # Find all boxes for possibel pieces
            all_boxes = all_bounding_boxes_2(warped, warped)
            # Check if there is a piece in each position of the chessboard
            board, number_pieces, boxes_chess = detect_pieces(warped)
            board = board.tolist()
            # With the all_boxes and the board pieces list identify the right boxes        
            best = detected_boxes(board, all_boxes, boxes_chess)
            # Transform the positions to original image 
            detected_pieces = perspective_to_original(H, best)
            detected_pieces = convert_to_dict_format(detected_pieces)
            # Find the inicial corner and rotate the image and the board list
            darkest_corner = find_darkest_corner(warped)
            board, corrected_image  = rotate_to_bottom_left(warped, board, darkest_corner)
            # Create output in the requested JSON format
            json_output(path, number_pieces, board, detected_pieces)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if the input file was provided
    if len(sys.argv) != 2:
        print("Usage: python3 chess_table.py input.json")
        sys.exit(1)
    
    input_json = sys.argv[1]
    main(input_json)
