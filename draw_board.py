import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess image
img = cv2.imread("output_images9/output_image_13.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Sobel filter (detect edges in both directions)
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

# Threshold to get binary image
_, thresh = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

# Morphology to close small gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# --- Detect grid size and divide into cells ---
height, width = morph.shape
cell_h, cell_w = height // 8, width // 8

# --- Detect pieces in each square ---
board_matrix = np.zeros((8, 8), dtype=int)

# Define percentage of inner square to analyze (e.g. 50% of the center)
inner_scale = 0.5
offset_h = int((1 - inner_scale) / 2 * cell_h)
offset_w = int((1 - inner_scale) / 2 * cell_w)
pieces=0
for row in range(8):
    for col in range(8):
        # Coordinates of the full square
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        
        # Crop to the central region to avoid grid lines
        cy1 = y1 + offset_h
        cy2 = y2 - offset_h
        cx1 = x1 + offset_w
        cx2 = x2 - offset_w
        central_cell = morph[cy1:cy2, cx1:cx2]
        
        # Count white pixels (edges) in the central region
        white_pixels = cv2.countNonZero(central_cell)
        if white_pixels > 300:  # Adjust threshold as needed
            board_matrix[row, col] = 1
            pieces+=1

# Print board matrix
print("Board Matrix (1 = piece, 0 = empty):")
for row in board_matrix:
    print(" ".join(str(val) for val in row))
print("number of pieces: ",f"{pieces}")

# Optional: visualize the result
plt.figure(figsize=(10, 10))
plt.imshow(morph, cmap='gray')
plt.title("Linhas detectadas com Sobel")
plt.axis("off")
plt.show()


