# Task 2 — Chess Piece Counting

This task estimates **how many chess pieces are on the board** in a given image, using
the [ChessRed2k](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f) dataset
(`chessred/`, indexed by `annotations.json`). The piece count (0–32) is predicted from
the raw image, treating the problem as a **regression / bounded-classification task**
rather than detecting individual pieces.

## Approach

All experiments are implemented in `piece_detection.py`, which trains and evaluates a
`ChessPieceCountDataset` (image → total piece count) with different backbones:

- **ConvNeXt-Tiny** and **VGG19** (pretrained, fine-tuned) — `best_model_convnext_tiny_classi.pth`, `best_model_vgg19_classi.pth`
- **Modified baseline CNN** (trained from scratch) — `best_modified_baseline_classi.pth`
- **YOLO-like ConvNeXt-Tiny**: the image is split into a grid and the CNN is applied
  per-cell in a YOLO-inspired way, instead of regressing on the whole image at once —
  `best_model_yolo_convnext_tiny.pth`

Optional flags at the top of the script control which experiment is run: fine-tuning the
full pretrained backbone vs. just the head (`BASE`), testing different input image sizes
for the from-scratch CNN (`SIZE`), comparing the two best pretrained models against the
scratch CNN (`CLASSI`), cropping images to the piece bounding boxes (`USE_CROP`), and the
grid-based YOLO-like variant (`YOLO_LIKE`).

## Evaluation

Predictions are rounded to the nearest integer piece count and compared against ground
truth using:
- **MAE**, **RMSE**, **R²**
- **Accuracy** (exact match) and **accuracy with ±1 piece tolerance**

## Dependencies

The program was developed with Python 3.12. To run the project, you need to install the following dependencies:

```bash
pip install torch torchvision
pip install pillow
pip install numpy
pip install scikit-learn
pip install tqdm
```

## How to run

- **Download Models**
    Download the best models using this [link](https://drive.google.com/drive/folders/1aFbZSyAQH90F6GMMkB4ji3Br-8F3JtZB?usp=sharing). Place each model in the same folder as the Python file.

- **Evaluate Results**
    To evaluate the results, execute the following command:

    ```bash
    python piece_detection.py <input.json> --model {best,yolo}
    ```

    - If the `--model` argument is not provided or is set to `best`, the script will use our best-performing model (ConvNeXt-Tiny/VGG19 based).

    - If the `--model` argument is set to any value other than `best`, the script will use our YOLO-like ConvNeXt-Tiny model instead.

- **Other models**
    Additional models used in this approach can be found using this [link](https://drive.google.com/drive/folders/1AoCPdrP-sQaQ5-RGTlmFc9fT5ysqES36?usp=sharing)
