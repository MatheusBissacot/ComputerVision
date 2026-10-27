# Task 3 — Chess Piece and Board Detection (Digital Twin)

This task builds a full pipeline that turns a photo of a chess game into a **digital
twin**: it detects the pieces, classifies their type and color, locates the board, and
reconstructs the position as a text/board representation. Everything is implemented in
[`Task3.ipynb`](Task3.ipynb), trained on the
[ChessRed2k](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f) dataset
(`images/`, `labels/`, `annotations.json`, with `data.yaml`/`data2.yaml` as the
Ultralytics YOLO dataset configs; trained runs are saved under `runs/detect`).

## 3.1 — Chess Piece Detection

Two detector families were compared, first just detecting *"piece vs. background"*:

- **Faster R-CNN** (ResNet-50 + FPN backbone, pretrained on COCO): the dataset was
  converted to COCO format and the model fine-tuned for 50 epochs. Full fine-tuning
  outperformed freezing the backbone and only training the detection head
  (mAP@[0.5:0.95] of **0.8032** vs. **0.7881**).
- **YOLOv8/YOLO11**, both pretrained and trained from scratch: the dataset was converted
  to the Ultralytics YOLO label format. Both variants reached a near-perfect
  **mAP@[0.5:0.95] ≈ 0.995**, with the pretrained model producing 6290 true positives, 4
  false negatives and 20 false positives on the test set (from-scratch: 6286 TP, 8 FN, 70
  FP) — confirming YOLO as the stronger and faster option overall.

## 3.2 — Board and Piece Type Prediction

The task was then extended from binary piece detection to full **piece classification**
(e.g. `white-queen`, `black-king`, ...) plus **board localization**:

- A YOLO model was retrained to predict the actual piece classes instead of a single
  "chess piece" label, reaching high precision (>0.90) across almost all classes, with
  further gains from higher input resolution and stronger data augmentation.
- The model was then extended again to also regress the **four corners of the board** as
  detection targets (rather than treating the whole board as one box), which is what
  allows the pieces to later be mapped onto a normalized board grid. Adding this task did
  not meaningfully hurt piece-classification performance.

## 3.3 — Digital Twin Reconstruction

Using the piece and corner predictions, the notebook builds the actual board state:

1. The predicted board corners are used to perspective-warp the image (and the piece
   bounding boxes) into a top-down, normalized view of the board.
2. When a corner is occluded by a piece, it is estimated as the far corner of the piece
   sitting on that position — this consistently happens at the top-right corner in this
   dataset.
3. The warped board is split into an 8x8 grid, and each detected piece is assigned to a
   square based on its warped position.
4. The reconstructed board is printed as text (uppercase = white pieces, lowercase =
   black pieces, dots = empty squares), correctly oriented so the orientation marker ends
   up in the bottom-left corner, matching standard chess board notation.

### Final results

Evaluated end-to-end over the test images, the reconstructed digital twin achieves
**85% accuracy**, with precision/recall/F1 all at **85%** (weighted average). Performance
tracks how frequent each piece type is in the training data: well-represented pieces like
black pawns (F1 = 0.90) and white rooks (F1 = 0.87) are recognized very reliably, while
rarer pieces such as the black king (F1 = 0.68) and white queen (F1 = 0.62) are
comparatively harder, highlighting class imbalance as the main remaining limitation.

## Dependencies

The project was developed using Python 3.12. To run it, install the required dependencies:

```bash
pip install torch torchvision
pip install pillow
pip install numpy
pip install scikit-learn
pip install tqdm
pip install ultralytics
```

> Note: `ultralytics` is required to run the YOLO models.

## How to run

Open and run [`Task3.ipynb`](Task3.ipynb) top to bottom. It expects the dataset to be
laid out as in `images/`, `labels/` and `annotations.json`, and uses `data.yaml`
(piece detection) / `data2.yaml` (piece type + board corners) as the YOLO dataset configs.
Training runs and checkpoints are written under `runs/detect/`.

Pretrained checkpoints used throughout this task can also be downloaded here:
[Best Models](https://drive.google.com/drive/folders/1aFbZSyAQH90F6GMMkB4ji3Br-8F3JtZB?usp=sharing) /
[Additional Models](https://drive.google.com/drive/folders/1AoCPdrP-sQaQ5-RGTlmFc9fT5ysqES36?usp=sharing)

## Final Note

This task demonstrates the potential of combining precise object detection with spatial
awareness (board corner detection + perspective warping) to digitize a real-world
chessboard, forming a foundation for game tracking, analysis, and AI interaction.
