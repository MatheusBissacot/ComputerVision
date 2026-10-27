# ComputerVision

Computer Vision course project (MSc in Artificial Intelligence): a set of tasks that
progressively build a "digital twin" system for chess games from photos of a physical
board, using the
[ChessRed2k dataset](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2).

## Tasks

- **[Task 1](Task%201)** — Classical computer vision baseline (`chess_table.py`): detects
  the chessboard in an image using OpenCV (edge/contour detection and perspective
  transform), then outputs the number of pieces, a binary occupancy matrix, and each
  piece's bounding box, without any learned model.
- **[Task 2](Task%202)** — Chess piece **counting**: given a board image, predicts how
  many pieces (0–32) are on it, framed as a regression/bounded-classification problem
  solved with CNN backbones (ConvNeXt-Tiny, VGG19, a custom CNN, and a YOLO-inspired
  grid-based variant). See [Task 2/README.md](Task%202/README.md).
- **[Task 3](Task%203)** — Full **piece detection, classification, board localization,
  and digital twin reconstruction**: compares Faster R-CNN and YOLO for piece detection
  (mAP@[0.5:0.95] up to 0.995), extends YOLO to classify piece type/color and locate the
  board corners, then warps the image to reconstruct the board state as text
  (85% end-to-end accuracy). See [Task 3/README.md](Task%203/README.md).

## References

- [Chess Recognizition Dataset](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2)
- [Chess Board OpenCV Recognition](https://github.com/doleron/chess-board-opencv-recognition)
- [Extracting Chess Square Coordinates Dynamically with OpenCV Image Processing Methods](https://medium.com/@siromermer/extracting-chess-square-coordinates-dynamically-with-opencv-image-processing-methods-76b933f0f64e)
- [Decoding Chess Boards (Video)](https://www.youtube.com/watch?v=7zLDhua9dQs)
- [Decoding Chess Boards](https://github.com/Elucidation/ChessboardDetect?tab=readme-ov-file#chessboard-detection)