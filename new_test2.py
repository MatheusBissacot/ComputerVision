import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import math


imagesDir = './'
outputDir = './output_images2/'  


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
    try:
        img = cv2.imread(img)
    except:
        img=img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)

    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    

    sobel_edges = cv2.magnitude(sobel_x, sobel_y)


    sobel_edges = cv2.convertScaleAbs(sobel_edges)


    ret, otsu_binary = cv2.threshold(sobel_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_clean = np.ones((3, 3), np.uint8)
    otsu_clean = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel_clean)

    canny = cv2.Canny(otsu_clean,100,255)

    kernel = np.ones((7, 7), np.uint8) 
    img_dilation = cv2.dilate(canny, kernel, iterations=1) 

    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=50)
    '''
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            cv2.line(img_dilation, (x1, y1), (x2, y2), (255, 255, 255), 2)

    kernel = np.ones((3, 3), np.uint8) 
    img_dilation_2 = cv2.dilate(img_dilation, kernel, iterations=1) 

    # find contours --> img_dilation_2
    board_contours, hierarchy = cv2.findContours(img_dilation_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_centers=list()

    # draw filtered rectangles to "canny" image for better visualization
    board_squared = canny.copy()  

    for contour in board_contours:
        if 4000 < cv2.contourArea(contour) < 20000:
            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the approximated contour has 4 points (quadrilateral)
            if len(approx) == 4:
                pts = [pt[0] for pt in approx]  # Extract coordinates

                # Define the points explicitly
                pt1 = tuple(pts[0])
                pt2 = tuple(pts[1])
                pt4 = tuple(pts[2])
                pt3 = tuple(pts[3])

                x, y, w, h = cv2.boundingRect(contour)
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                square_centers.append([center_x,center_y,pt2,pt1,pt3,pt4])

                # Draw the lines between the points
                cv2.line(board_squared, pt1, pt2, (255, 255, 0), 7)
                cv2.line(board_squared, pt1, pt3, (255, 255, 0), 7)
                cv2.line(board_squared, pt2, pt4, (255, 255, 0), 7)
                cv2.line(board_squared, pt3, pt4, (255, 255, 0), 7)


    sorted_coordinates = sorted(square_centers, key=lambda x: x[1], reverse=True)

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

    print(sorted_coordinates[:10])

    for coord in sorted_coordinates:
        center_x, center_y = int(coord[0]), int(coord[1])  # Converte para inteiros
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  # Desenha o ponto vermelho
    '''
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
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    return black_image, img

def find_board_contour(binary_image):
    """
    binary_image: imagem preta com linhas brancas (já binarizada)
    Retorna os 4 cantos do maior quadrado detetado.
    """
    
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_image, cmap='gray')
    plt.title("Piece detection")
    plt.axis("off")
    plt.show()
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        # Aproximação do contorno
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and area > max_area:
            max_area = area
            best_cnt = approx

    if best_cnt is not None:
        pts = best_cnt.reshape(4, 2)
        rect = order_points(pts)
        return rect
    else:
        return None


def order_points(pts):
    # Ordena pontos no formato: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect

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
        
    plt.title("Geometrically possible valid squares on original image \n squares_image")
    plt.imshow(squares_image,cmap="gray")
    plt.show()

    plt.title("All squares \n all_contours_image")
    plt.imshow(all_contours_image,cmap="gray")
    plt.show()

    return sorted_coordinates
 
def transform_prespective(rect):
    if rect is not None:
        width, height = 800, 800  # tamanho final desejado do tabuleiro
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        H = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, H, (width, height))
    return warped, H

def prespective_to_original(H, positions):
    H_inv = np.linalg.inv(H)
    original_positions = []

    for pos in positions:
        pos = pos[2:]
        pos = np.array(pos, dtype=np.float32)
        pos = pos.reshape(1, -1, 2)
        transformed = cv2.perspectiveTransform(pos, H_inv)
        original_positions.append(transformed)

    return original_positions


def detect_pieces(warped, positions=None, margin=50): # melhorar a deteção das peças
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

    # Convert ROI to RGB for visualization
    roi_rgb = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2RGB)

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            cy1 = y1 + offset_h
            cy2 = y2 - offset_h
            cx1 = x1 + offset_w
            cx2 = x2 - offset_w

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

    # Show processed ROI with white pixel counts
    plt.figure(figsize=(10, 10))
    plt.imshow(roi_rgb)
    plt.title("White Pixel Counts in Grid Cells")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(morph, cmap='gray')
    plt.title("Enhanced Piece Detection (Dilated Edges)")
    plt.axis("off")
    plt.show()

    return board_matrix, pieces

def json_output(input_image, number_pieces, board, detected_pieces):
    # Estrutura de dados inicial
    data = [
        {
            "image": input_image,
            "num_pieces": number_pieces,
            "board": board,
            "detected_pieces": detected_pieces
        }
    ]
    
    # Salvar em um arquivo (opcional)
    with open('output.json', 'w') as file:
        json.dump(data, file)
 
def read_images(paths):
    for i, p in enumerate(paths):
       
        output_name = f"output_image_{i + 1}.png"
        transform_image(p, output_name)



all_paths = json_to_path(os.path.join(imagesDir, 'image_paths.json'))

#Transforma a imagem

black_image, image = transform_image(all_paths[8], 'a')



#Encontra o tabuleiro
rect = find_board_contour(black_image)

print(rect)

#Coloco o tabuleiro numa prespectiva central
warped, H = transform_prespective(rect)

#Transforma o tabuleiro em prespetiva
black_image2, image2 = transform_image(warped, 'a')

#Identifica cada um dos quadrados do tabuleiro
#pos = validate_squares(black_image2, image2)

#Verificar se tem peça em cada posição do tabuleiro
board, number_pieces = detect_pieces(warped)
for row in board:
    print(" ".join(str(val) for val in row))
print("number of pieces: ",f"{number_pieces}")


board = board.tolist()
#Transforma as posiçoes da imagem em prespetiva em posiçoes na imagem original dos quadrados que tem peça
#detected_pieces = prespective_to_original(H, pos) #Falta colocar apenas nos quadrados que tem peça e colocar no formato pedido
#print(detected_pieces)

detected_pieces = 0 #Remover isto
#Criar output no formato json pedido
json_output(all_paths[0], number_pieces, board, detected_pieces)