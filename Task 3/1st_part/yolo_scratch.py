'''
Got it — you're building a YOLOv1-style model from scratch to detect chess pieces on a board, and you want a full implementation with:

Realistic YOLOv1 structure

Custom loss function (with confidence, coordinates)

Post-processing: IoU, Non-Maximum Suppression (NMS)

mAP metric (mean average precision)

Code structured to be extensible for real datasets

✅ This version assumes you’ll use your own chess piece dataset formatted to return:

image (Tensor[3 x H x W])

target (Tensor[S, S, 5]) → [x_center, y_center, w, h, confidence]


To make this work with real chess piece images:

Create a dataset with:

image → resized to 224x224

Labels in YOLO format: S x S x 5 per image

Replace ChessPieceDataset with your custom dataset loader.

Improve visual post-processing with bounding boxes and class names.
'''
# yolo_chess.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ------------------- Dataset -------------------
class ChessPieceDataset(Dataset):
    def __init__(self, num_samples=500):
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.zeros(num_samples, 7, 7, 5)
        # Random boxes in YOLO format (x, y, w, h, conf)
        for i in range(num_samples):
            cell_x, cell_y = np.random.randint(0, 7), np.random.randint(0, 7)
            self.labels[i, cell_y, cell_x] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ------------------- Model -------------------
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=1):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, padding=1), nn.BatchNorm2d(192), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 496), nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (B * 5)),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x.view(-1, self.S, self.S, 5)

# ------------------- Loss -------------------
def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5):
    obj_mask = target[..., 4] > 0
    noobj_mask = target[..., 4] == 0
    mse = nn.MSELoss()

    # Coord loss (x, y)
    coord_loss = mse(pred[obj_mask][..., 0:2], target[obj_mask][..., 0:2])
    # Coord loss (sqrt w, h)
    coord_loss += mse(torch.sqrt(pred[obj_mask][..., 2:4] + 1e-6),
                      torch.sqrt(target[obj_mask][..., 2:4] + 1e-6))

    # Confidence loss
    obj_loss = mse(pred[obj_mask][..., 4], target[obj_mask][..., 4])
    noobj_loss = mse(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])

    return lambda_coord * coord_loss + obj_loss + lambda_noobj * noobj_loss

# ------------------- IoU -------------------
def compute_iou(box1, box2):
    # box = [x_center, y_center, w, h]
    box1 = box1.clone()
    box2 = box2.clone()

    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4] / 2
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4] / 2

    box1_min = box1_xy - box1_wh
    box1_max = box1_xy + box1_wh
    box2_min = box2_xy - box2_wh
    box2_max = box2_xy + box2_wh

    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    union = area1 + area2 - inter_area

    return inter_area / (union + 1e-6)

# ------------------- mAP Metric -------------------
def calculate_map(preds, targets, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    for pred, target in zip(preds, targets):
        pred_boxes = pred[pred[..., 4] > 0.5]
        target_boxes = target[target[..., 4] > 0.5]

        matched = []
        for t in target_boxes:
            best_iou = 0
            best_idx = -1
            for i, p in enumerate(pred_boxes):
                if i in matched:
                    continue
                iou = compute_iou(p.unsqueeze(0), t.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh:
                TP += 1
                matched.append(best_idx)
            else:
                FN += 1
        FP += len(pred_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return precision, recall, TP, FP, FN

# ------------------- Training -------------------
def train_yolo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv1().to(device)
    dataset = ChessPieceDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        preds_all, targets_all = [], []
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            pred = model(imgs)

            loss = yolo_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds_all.extend(pred.detach().cpu())
            targets_all.extend(targets.cpu())

        precision, recall, TP, FP, FN = calculate_map(preds_all, targets_all)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | mAP@0.5 → P: {precision:.3f}, R: {recall:.3f}, TP: {TP}, FP: {FP}, FN: {FN}")

if __name__ == "__main__":
    train_yolo()
