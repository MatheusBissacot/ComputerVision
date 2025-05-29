import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# ---------- Config ----------
JSON_PATH = "annotations.json"
IMAGE_ROOT = "."  # Adjust if images are nested
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# ---------- Dataset Class ----------
class ChessPieceCountDataset(Dataset):
    def __init__(self, json_path, split='train', transform=None, image_root=''):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform
        self.image_root = image_root
        self.image_ids = self.data['splits'][split]['image_ids']
        self.image_dict = {img['id']: img for img in self.data['images']}

        self.count_map = {}
        for piece in self.data['annotations']['pieces']:
            img_id = piece['image_id']
            self.count_map[img_id] = self.count_map.get(img_id, 0) + 1

        self.samples = []
        for img_id in self.image_ids:
            img_meta = self.image_dict[img_id]
            file_path = os.path.join(image_root, 'chessred', img_meta['path'])
            count = self.count_map.get(img_id, 0)
            self.samples.append((file_path, count))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, count = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([count], dtype=torch.float32)

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- CNN from Scratch ----------
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # 224x224 → 112x112
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # Downsampling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 112x112 → 56x56
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 56x56 → 28x28

        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 64, 28, 28)
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)                       # Output: (B, 1)
        return x


# ---------- Model ----------
def get_model(model_name="resnet18"):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 1)

    elif model_name == "baseline":
        model = BaselineCNN()

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model



# ---------- Train ----------
def train(model, train_loader, val_loader):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

        if val_loader:
            evaluate(model, val_loader)

    torch.save(model.state_dict(), "model.pth")

# ---------- Evaluate ----------
def evaluate(model, loader):
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).cpu().numpy().flatten()
            y_pred.extend(outputs)
            y_true.extend(targets.numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")

# ---------- Predict ----------
def predict(image_path, model_path="model.pth"):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
    count = output.item()
    print(f"Predicted piece count: {count:.2f}")
    return count

# ---------- Main ----------
if __name__ == "__main__":
    # Load datasets
    train_set = ChessPieceCountDataset(JSON_PATH, split='train', transform=transform, image_root=IMAGE_ROOT)
    val_set = ChessPieceCountDataset(JSON_PATH, split='val', transform=transform, image_root=IMAGE_ROOT)
    test_set = ChessPieceCountDataset(JSON_PATH, split='test', transform=transform, image_root=IMAGE_ROOT)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Train and evaluate
    # # Training with Resnet18
    # model = get_model("resnet18")
    # Training with VGG16
    model = get_model("vgg16")
    train(model, train_loader, val_loader)

    # Final evaluation on test set
    print("Test evaluation:")
    evaluate(model, test_loader)

    # Example prediction
    # predict("images/0/G000_IMG000.jpg")