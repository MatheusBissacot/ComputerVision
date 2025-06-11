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
from torch.optim.lr_scheduler import StepLR

# ---------- Config ----------
JSON_PATH = "../annotations.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
TASK_TYPE = "regression"  # "classification" 
MAX_PIECES = 33  # 0 to 32 pieces

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
            
        if TASK_TYPE == "regression":
            target = torch.tensor([count], dtype=torch.float32)
        else:
            target = torch.tensor(count, dtype=torch.long)
        
        return image, target

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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes_or_output=33, task_type="classification"):
        super(CNN, self).__init__()

        self.task_type = task_type
        output_dim = num_classes_or_output if task_type == "classification" else 1

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output (B, 512, 1, 1)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(0.5)

        # Skip connections: simple 1x1 conv to match dimensions where needed
        self.skip3 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # H/2, W/2

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # H/4, W/4

        # Block 3 with residual connection
        identity3 = self.skip3(x)
        out3 = F.relu(self.conv3_1(x))
        out3 = F.relu(self.bn3(self.conv3_2(out3)))
        out3 += identity3
        out3 = F.max_pool2d(out3, 2)  # H/8, W/8

        # Block 4 with residual connection
        identity4 = self.skip4(out3)
        out4 = F.relu(self.conv4_1(out3))
        out4 = F.relu(self.bn4(self.conv4_2(out4)))
        out4 += identity4
        out4 = F.max_pool2d(out4, 2)  # H/16, W/16

        # Global pooling
        out = self.global_avg_pool(out4)  # (B, 512, 1, 1)
        out = out.view(out.size(0), -1)   # (B, 512)

        # Fully connected head with dropout
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.out(out)

        return out




# ---------- Model ----------
def get_model(model_name):
    output_dim = MAX_PIECES if TASK_TYPE == "classification" else 1
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_dim)

    elif model_name == "baseline":
        model = CNN()

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_dim)
    
    elif model_name == "vgg19_512":
        base = models.vgg19(pretrained=True)
        model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # converte para (B, 512, 1, 1)
            nn.Flatten(),                 # converte para (B, 512)
            nn.Linear(512, 1)             # regressão
        )
        
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_dim)
    
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, output_dim)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model



# ---------- Train ----------
def train(model, train_loader, val_loader, patience=5, fine_tune_epochs=5):
    model = model.to(DEVICE)
    
    if TASK_TYPE == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Split model into feature extractor and classifier
    if hasattr(model, 'features'):  # e.g. VGG, EfficientNet, ConvNeXt
        features = model.features
        classifier = model.classifier
    elif hasattr(model, 'backbone'):  # Some custom or ConvNeXt models
        features = model.backbone
        classifier = model.classifier
    elif isinstance(model, models.ResNet) or hasattr(model, 'fc'):
        # For ResNet-like
        features = nn.Sequential(*(list(model.children())[:-1]))
        classifier = model.fc
    else:
        features = model
        classifier = None

    # -------- Phase 1: Freeze base & train only classifier --------
    if classifier:
        for param in features.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True  # If baseline or custom

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

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

        scheduler.step()

        print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

        # ---------- Validation ----------
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

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Best model saved with Validation Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # ---------- Unfreeze and fine-tune full model ----------
        if epoch + 1 == fine_tune_epochs:
            print("Unfreezing full model for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)  # smaller LR
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


    

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
            outputs = model(imgs)
            
            if TASK_TYPE == "classification":
                preds = outputs.argmax(dim=1).cpu().numpy()
                true_vals = targets.numpy()
            else:
                preds = outputs.cpu().numpy().flatten()
                true_vals = targets.numpy().flatten()
                
            y_pred.extend(preds)
            y_true.extend(true_vals)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    accuracy_tol = accuracy_with_tolerance(y_true,  np.round(y_pred), tolerance)
    accuracy = np.mean(np.array(y_true) == np.round(y_pred))

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f} | Accuracy (±{tolerance}): {accuracy_tol:.2%} | Accuracy: {accuracy:.2%}")

# ---------- Predict ----------
def predict(image_path, model_name="convnext_tiny", model_path="best_model_convnext_tiny.pth"):
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # Apply same transform used during training
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
    
    # Make sure the count is at least zero (no negative piece counts)
    count = max(0.0, float(output))
    count_rounded = round(count)
    print(f"Predicted piece count: {count:.2f} (rounded: {count_rounded})")
    return count_rounded

# ---------- Main ----------

#usar no efficient net b2
#transform_b2 = transforms.Compose([
#    transforms.Resize((260, 260)),
#    transforms.RandomHorizontalFlip(),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#])


if __name__ == "__main__":
    # Load datasets
    train_set = ChessPieceCountDataset(JSON_PATH, split='train', transform=transform)
    val_set = ChessPieceCountDataset(JSON_PATH, split='val', transform=transform)
    test_set = ChessPieceCountDataset(JSON_PATH, split='test', transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Train and evaluate
    
    # Training with VGG19
    # model = get_model("vgg19")
    
    # Training with Convnext_tiny
    #model = get_model("convnext_tiny")
    
    # Training from scratch
    model= get_model("baseline")
    
    #train(model, train_loader, val_loader)

    # Final evaluation on test set
    
    model.load_state_dict(torch.load("best_model_modified_baseline_regr.pth"))
    model.to(DEVICE)
    print("Test evaluation:")
    evaluate(model, test_loader)

    # Example prediction
    #predict("G000_IMG000.jpg")
