import cv2
import numpy as np
import os
import json

# === Setup Paths ===
imagesDir = './'
outputDir = './output_images9/'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# === Load Image Paths from JSON ===
def json_to_path(json_file):
    path_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for row in data["images"]:
            path_list.append(row["path"])
    return path_list

# === Order the 4 points (used for perspective transform) ===
def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

# === Detect and Extract Top-Down Board ===
def transform_image(img_path, output_name):
    img = cv2.imread(img_path)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            board_contour = approx
            break
    else:
        print(f"No board found in {img_path}")
        cv2.imwrite(os.path.join(outputDir, output_name), img)
        return

    rect = order_points(board_contour)
    dst_size = 800
    dst_pts = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(orig, M, (dst_size, dst_size))

    # Save the top-down view
    output_path = os.path.join(outputDir, output_name)
    cv2.imwrite(output_path, warped)
    print(f"Saved top-down view: {output_name}")

    # Detect and draw pieces
    #detect_pieces_on_board(warped)

def detect_pieces_on_board(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    board_matrix = [[0 for _ in range(8)] for _ in range(8)]

    cell_h = warped_img.shape[0] // 8
    cell_w = warped_img.shape[1] // 8

    output_img = warped_img.copy()

    for i in range(8):
        for j in range(8):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            cell = gray[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(cell, (5, 5), 0)

            # Adaptive thresholding + remove lines
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Morphological closing to fill gaps (remove noise)
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            piece_found = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 80:  # too small, ignore noise
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w < 5 or h < 5:
                    continue

                # Assume valid piece if it covers enough space
                if area > 120:
                    board_matrix[i][j] = 1
                    cv2.rectangle(output_img, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)
                    piece_found = True
                    break

            if not piece_found:
                board_matrix[i][j] = 0


    # Save annotated image
    cv2.imwrite(os.path.join(outputDir, "with_boxes.png"), output_img)
    return board_matrix


# === Run on All Images in the List ===
def read_images(paths):
    for i, p in enumerate(paths):
        output_name = f"output_image_{i + 1}.png"
        transform_image(p, output_name)

# === MAIN ===
all_paths = json_to_path(os.path.join(imagesDir, 'image_paths.json'))
read_images(all_paths)
