# yolo_pretrained.py
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.ops import box_iou, nms
from ultralytics import YOLO  # Requires: pip install ultralytics

# --------------- Dataset ----------------
class DummyChessDataset:
    def __init__(self, root="images/"):
        self.image_paths = list(Path(root).glob("*.jpg"))
        self.boxes = [torch.tensor([[50, 60, 120, 160]], dtype=torch.float32) for _ in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = F.to_tensor(img)
        return img_tensor, self.boxes[idx]

# --------------- Metrics ----------------
def evaluate_map(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    for preds, scores, targets in zip(pred_boxes, pred_scores, gt_boxes):
        preds = preds[scores > 0.5]
        matched = []
        for t in targets:
            best_iou = 0
            for i, p in enumerate(preds):
                if i in matched: continue
                iou = box_iou(p.unsqueeze(0), t.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh:
                TP += 1
                matched.append(best_idx)
            else:
                FN += 1
        FP += len(preds) - len(matched)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return precision, recall, TP, FP, FN

# --------------- Inference ----------------
def run_yolo_evaluation():
    model = YOLO("yolov5s.pt")  # Use yolov5n, yolov5m, etc. as needed
    model.fuse()

    dataset = DummyChessDataset()
    preds_all, scores_all, targets_all = [], [], []

    for i in range(len(dataset)):
        img, gt_box = dataset[i]
        results = model(img.permute(1, 2, 0).numpy(), verbose=False)[0]

        boxes = results.boxes.xyxy
        scores = results.boxes.conf
        if boxes.nelement() == 0:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
        preds_all.append(boxes.cpu())
        scores_all.append(scores.cpu())
        targets_all.append(gt_box)

    p, r, TP, FP, FN = evaluate_map(preds_all, scores_all, targets_all)
    print(f"YOLOv5 → P: {p:.3f}, R: {r:.3f}, TP: {TP}, FP: {FP}, FN: {FN}")

if __name__ == "__main__":
    run_yolo_evaluation()
