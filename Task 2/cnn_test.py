import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm

# ---------- Config ----------
JSON_PATH = "annotations.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# ---------- Dataset Class ----------
class ChessPieceCountDataset(Dataset):
    def __init__(self, json_path, split, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform
        self.image_ids = self.data['splits'][split]['image_ids']
        self.image_dict = {img['id']: img for img in self.data['images']}

        self.count_map = {}
        for piece in self.data['annotations']['pieces']:
            img_id = piece['image_id']
            self.count_map[img_id] = self.count_map.get(img_id, 0) + 1

        self.samples = []
        for img_id in self.image_ids:
            img_meta = self.image_dict[img_id]
            file_path = os.path.join('chessred', img_meta['path'])
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
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- CNN from Scratch ----------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 224x224 → Output: 224x224
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 224x224
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 112x112
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling: Halves dimensions
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutional Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 112, 112)
        
        # Convolutional Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 56, 56)
        
        # Convolutional Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B, 128, 28, 28)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 128 * 28 * 28)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)  # Output: (B, 1)
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
        
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 1)
        
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 1)
    
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model



# ---------- Train ----------
def train(model, train_loader, val_loader, patience=5):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Training loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for imgs, targets in loop:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation loop
        if val_loader:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for imgs, targets in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    preds = model(imgs)
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model_scratch.pth")
                print(f"Best model saved with Validation Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    

# ---------- Evaluate ----------

def accuracy_with_tolerance(y_true, y_pred, tolerance=1):
    correct = sum(abs(true - pred) <= tolerance for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)


def evaluate(model, loader, tolerance=1):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).cpu().numpy().flatten()
            y_pred.extend(outputs)
            y_true.extend(targets.numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    accuracy_tol = accuracy_with_tolerance(y_true, y_pred, tolerance)
    accuracy = np.mean(np.array(y_true) == np.round(y_pred))

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f} | Accuracy (±{tolerance}): {accuracy_tol:.2%} | Accuracy: {accuracy:.2%}")

# ---------- Predict ----------
def predict(image_path, model_path="model.pth"):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
    #count = output.item()
    count = max(0.0, float(output))
    print(f"Predicted piece count: {count:.2f}")
    return count

# ---------- Main ----------

#usar no efficient net b2
transform_b2 = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    # Load datasets
    train_set = ChessPieceCountDataset(JSON_PATH, split='train', transform=transform)
    val_set = ChessPieceCountDataset(JSON_PATH, split='val', transform=transform)
    test_set = ChessPieceCountDataset(JSON_PATH, split='test', transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Train and evaluate
    # Training with Resnet18
    #model = get_model("efficientnet_b2")
    # Training with VGG16
    #model = get_model("resnet34")
    #model = get_model("vgg16")
    model = CNN()
    train(model, train_loader, val_loader)

    # Final evaluation on test set
    
    model.load_state_dict(torch.load("best_model_scratch.pth"))
    model.to(DEVICE)
    #print("Test evaluation:")
    evaluate(model, test_loader)

    # Example prediction
    # predict("images/0/G000_IMG000.jpg")
