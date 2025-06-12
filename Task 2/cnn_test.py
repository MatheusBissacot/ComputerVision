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
import argparse

# ----------Arguments --------
parser = argparse.ArgumentParser()
parser.add_argument('--obj', default="test")
args = parser.parse_args()

# ---------- Config ----------
JSON_PATH = "annotations.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
TASK_TYPE = "regression"  # "classification" 
MAX_PIECES = 33  # 0 to 32 pieces for the classification task

# Additional Configurations
BASE = False  # fine tune all pretrain models
SIZE = False  # Test diferent image sizes on the CNN from scratch (pretrained were too computational expensive)
CLASSI = False  # Two best pretrained models chosen (1 with a higher number and 1 with lower number of parameters) and the CNN from scratch trained as a classification task
USE_CROP = False # Crop the images based on the bboxes delimitations (crop the images to focus more on the pieces than the background)
YOLO_LIKE = False  # Use the CNNs chosen in a similar way yolo models work

# ---------- Dataset Class ----------

class ChessPieceCountDataset(Dataset):
    def __init__(self, json_path, split, dataset, transform=None):
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
            img_meta = self.image_dict.get(img_id)
            if not img_meta:
                continue

            file_path = os.path.join(dataset, img_meta['path'])

            # Verify if the file exists
            if not os.path.exists(file_path):
                continue

            count = self.count_map.get(img_id, 0)
            self.samples.append((file_path, count))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        
        img_path, count = self.samples[idx]
            
        image = Image.open(img_path).convert("RGB")
        
        if USE_CROP :
            image = self.crop_board(image, img_path)       
            
        
        if self.transform:
            image = self.transform(image)
            
        if TASK_TYPE == "regression":
            target = torch.tensor([count], dtype=torch.float32)
        else:
            target = torch.tensor(count, dtype=torch.long)
        
        return image, target
    
    def crop_board(self, image, img_path):
    
        img_id = None
        for img in self.image_dict.values():
            if img['path'] in img_path:
                img_id = img['id']
                break

        if img_id is None:
            raise ValueError(f"No image found: {img_path}")

        bboxes = []
        for piece in self.data['annotations']['pieces']:
            
            if piece['image_id'] == img_id:
                bboxes.append(piece['bbox'])
                
        img_id = next(img['id'] for img in self.image_dict.values() if img['path'] in img_path)
        bboxes = [p['bbox'] for p in self.data['annotations']['pieces'] if p['image_id'] == img_id]
        if not bboxes:
            return image

        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[0] + b[2] for b in bboxes)
        y1 = max(b[1] + b[3] for b in bboxes)

        return image.crop((x0, y0, x1, y1))
        

# ---------- Transforms ----------

def diff_sizes(size):
    IMG_SIZE= size
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return transform

# ---------- CNN from Scratch ----------
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        
        output_dim = MAX_PIECES if TASK_TYPE == "classification" else 1
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  

        x = self.global_avg_pool(x)  
        x = x.view(x.size(0), -1)    

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    
# ---------- CNN from Scratch Imp ----------    
class CNN_imp(nn.Module):
    def __init__(self):
        super(CNN_imp, self).__init__()

        output_dim = MAX_PIECES if TASK_TYPE == "classification" else 1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(0.5)

        self.skip3 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) 

        identity3 = self.skip3(x)
        out3 = F.relu(self.conv3_1(x))
        out3 = F.relu(self.bn3(self.conv3_2(out3)))
        out3 = out3 + identity3
        out3 = F.max_pool2d(out3, 2) 

        identity4 = self.skip4(out3)
        out4 = F.relu(self.conv4_1(out3))
        out4 = F.relu(self.bn4(self.conv4_2(out4)))
        out4 = out4 + identity4
        out4 = F.max_pool2d(out4, 2) 

        out = self.global_avg_pool(out4)  
        out = out.view(out.size(0), -1)   

        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.out(out)

        return out
    
# ---------- CNN for Yolo ----

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = F.adaptive_avg_pool2d(x, (7, 7))            
        return x

# ---------- Yolo Idea ------

class YoloStyleCountModel(nn.Module):
    def __init__(self, backbone_name, grid_size=7, num_classes=MAX_PIECES):
        super().__init__()

        self.backbone_name = backbone_name
        self.task_type = TASK_TYPE
        self.grid_size = grid_size  
        self.num_classes = num_classes
        self.backbone_out_channels = None  

        if backbone_name == "vgg19":
            base = models.vgg19(pretrained=True)
            self.backbone = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d((grid_size, grid_size))  
            )
            self.backbone_out_channels = 512

        elif backbone_name == "convnext_tiny":
            base = models.convnext_tiny(pretrained=True)
            self.backbone = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d((grid_size, grid_size))  
            )
            self.backbone_out_channels = 768

        elif backbone_name == "Scratch_CNN":
            self.backbone = CNNBackbone()
            self.backbone_out_channels = 128

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.head = nn.Conv2d(self.backbone_out_channels, 1, kernel_size=1)  
        self.final = nn.Identity()  

    def forward(self, x):
        x = self.backbone(x)  
        x = self.head(x)      
        x = x.view(x.size(0), -1)  
        return x.sum(dim=1, keepdim=True)  

# ---------- Model ----------

def get_model(model_name, backbone):
    output_dim = MAX_PIECES if TASK_TYPE == "classification" else 1
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_dim)

    elif model_name == "Scratch_CNN":
        model = CNN()
        
    elif model_name == "Scratch_CNN_imp":
        model = CNN_imp()

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_dim)
        
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
    
    elif model_name == "yolo":
        model = YoloStyleCountModel(backbone)

    else:
       raise ValueError(f"Unsupported model: {model_name}")

    return model

# ---------- Train ----------

def train(model, train_loader, val_loader, name, patience):
    model = model.to(DEVICE)
    
    if TASK_TYPE == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Training loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in loop:
            imgs, targets = batch[:2]
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            preds = model(imgs)
            
            if YOLO_LIKE:
                if TASK_TYPE == "regression":
                    preds = preds.view(preds.size(0), -1).sum(dim=1)
                    targets = targets.float().squeeze(1)

                else:  # classification
                    preds_per_grid = preds
                    preds_classes = preds_per_grid.argmax(dim=2) 
                    preds_sum = preds_classes.sum(dim=1).float()  
                    preds = preds_sum.unsqueeze(1) 
                    targets = targets.float()
        
            else:
                if TASK_TYPE == "regression":
                    targets = targets.float()
                else:
                    targets = targets.long()

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
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, targets = batch[:2]
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    
                    preds = model(imgs)
                    
                    if YOLO_LIKE:
                        if TASK_TYPE == "regression":
                            preds = preds.view(preds.size(0), -1).sum(dim=1)
                            targets = targets.float().squeeze(1)

                        else:  
                            preds_per_grid = preds 
                            preds_classes = preds_per_grid.argmax(dim=2)  
                            preds_sum = preds_classes.sum(dim=1).float() 
                            preds = preds_sum.unsqueeze(1)  
                            targets = targets.float()
                    else:
                        if TASK_TYPE == "regression":
                            targets = targets.float()
                        else:
                            targets = targets.long()
                            
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), name+".pth")
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
            outputs = model(imgs)
            
            if YOLO_LIKE:
                if TASK_TYPE == "regression":
                    outputs = outputs.view(outputs.size(0), -1).sum(dim=1)
                    preds = outputs.cpu().numpy().flatten()
                    true_vals = targets.numpy().flatten()
                else:
                    preds_classes = outputs.argmax(dim=2)  
                    preds_sum = preds_classes.sum(dim=1).cpu().numpy() 
                    preds = preds_sum
                    true_vals = targets.numpy()
            else:
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

# ---------- Main ----------
"""
Run cnn_test --obj <<objective>>   # objective must be changed to train or test, if ommited it will run the default value "test"    
"""

if __name__ == "__main__":
    # Load datasets
    train_set = ChessPieceCountDataset(JSON_PATH, split='train',dataset='chessred', transform=diff_sizes(224))
    val_set = ChessPieceCountDataset(JSON_PATH, split='val', dataset='chessred', transform=diff_sizes(224))
    test_set = ChessPieceCountDataset(JSON_PATH, split='test',dataset='chessred', transform=diff_sizes(224))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    

    if (args.obj == "train"):
        
        if BASE:
            mods=["vgg19", "convnext_tiny", "Scratch_CNN_imp", "resnet18", "vgg16", "efficientnet_b0", "vgg19_bn", "resnet34"] 
            for name in mods:
                model= get_model(name, "")
                train(model, train_loader, val_loader, "best_model_" + name, 5)
                
                # Evaluation
                model.load_state_dict(torch.load("best_model_" + name+ ".pth"))
                model.to(DEVICE)
                print(name)
                evaluate(model, test_loader)
            
            # EfficientNet_b2 needs a bigger image size (260,260)
            train_set2 = ChessPieceCountDataset(JSON_PATH, split='train', dataset='chessred', transform=diff_sizes(260))
            val_set2 = ChessPieceCountDataset(JSON_PATH, split='val', dataset='chessred', transform=diff_sizes(260))
            test_set2 = ChessPieceCountDataset(JSON_PATH, split='test', dataset='chessred', transform=diff_sizes(260))

            train_loader2 = DataLoader(train_set2, batch_size=BATCH_SIZE, shuffle=True)
            val_loader2 = DataLoader(val_set2, batch_size=BATCH_SIZE)
            test_loader2 = DataLoader(test_set2, batch_size=BATCH_SIZE)
            
            model= get_model("efficientnet_b2", "")
            train(model, train_loader2, val_loader2, "best_model_" + "efficientnet_b2", 5)
            
            # Evaluation
            model.load_state_dict(torch.load("best_model_efficientnet_b2.pth"))
            model.to(DEVICE)
            print("efficientnet_b2")
            evaluate(model, test_loader2)      
                
        if SIZE:
            for s in [224, 512, 1024]: 
                train_set_size = ChessPieceCountDataset(JSON_PATH, split='train', dataset='chessred', transform=diff_sizes(s))
                val_set_size = ChessPieceCountDataset(JSON_PATH, split='val', dataset='chessred', transform=diff_sizes(s))
                test_set_size = ChessPieceCountDataset(JSON_PATH, split='test', dataset='chessred', transform=diff_sizes(s))

                train_loader_size = DataLoader(train_set_size, batch_size=BATCH_SIZE, shuffle=True)
                val_loader_size = DataLoader(val_set_size, batch_size=BATCH_SIZE)
                test_loader_size = DataLoader(test_set_size, batch_size=BATCH_SIZE)
                
                model= get_model("Scratch_CNN", "")
                train(model, train_loader_size, val_loader_size, "best_model_Scratch_CNN_" + str(s), 5)
                
                # Evaluation
                model.load_state_dict(torch.load("best_model_Scratch_CNN_" + str(s) + ".pth"))
                model.to(DEVICE)
                print("best_model_Scratch_CNN_" + str(s))
                evaluate(model, test_loader_size)  
                
                if s == 512:
                    
                    model= get_model("convnext_tiny", "")
                    train(model, train_loader, val_loader, "best_model_convnext_tiny_512",5)
                            
                    # Evaluation
                    model.load_state_dict(torch.load("best_model_convnext_tiny_512.pth"))
                    model.to(DEVICE)
                    print("best_model_convnext_tiny_512.pth")
                    evaluate(model, test_loader)
                
        if CLASSI:
            TASK_TYPE = "classification"  # "classification"
            backbone=["vgg19", "convnext_tiny", "Scratch_CNN"]
            # Train
            for name in backbone:
                model= get_model(name, "")
                train(model, train_loader, val_loader, "best_model_" + name + "_classi",5)
                
                # Evaluation
                model.load_state_dict(torch.load("best_model_" + name + "_classi.pth"))
                model.to(DEVICE)
                print("best_model_" + name + "_classi.pth")
                evaluate(model, test_loader)
                
        if USE_CROP:
            train_set_crop = ChessPieceCountDataset(JSON_PATH, split='train',dataset='chessred2k', transform=diff_sizes(224))
            val_set_crop = ChessPieceCountDataset(JSON_PATH, split='val', dataset='chessred2k', transform=diff_sizes(224))
            test_set_crop = ChessPieceCountDataset(JSON_PATH, split='test',dataset='chessred2k', transform=diff_sizes(224))

            train_loader_crop = DataLoader(train_set_crop, batch_size=BATCH_SIZE, shuffle=True)
            val_loader_crop = DataLoader(val_set_crop, batch_size=BATCH_SIZE)
            test_loader_crop = DataLoader(test_set_crop, batch_size=BATCH_SIZE)
            
            mods=["vgg19", "convnext_tiny", "Scratch_CNN"] 
            # Train
            for name in mods:
                model= get_model(name, "")
                train(model, train_loader_crop, val_loader_crop, "best_model_" + name + "_crop",5) 
                
                # Evaluation
                model.load_state_dict(torch.load("best_model_" + name + "_crop.pth")) 
                model.to(DEVICE)
                print("best_model_" + name + "_crop")
                evaluate(model, test_loader_crop)
            
            TASK_TYPE = "classification"
            mods=["vgg19", "convnext_tiny", "Scratch_CNN"] 
            # Train
            for name in mods:
                model= get_model(name, "")
                train(model, train_loader_crop, val_loader_crop, "best_model_" + name + "_classi_crop.pth",5) 
                
                # Evaluation
                model.load_state_dict(torch.load("best_model_" + name + "_classi_crop.pth")) 
                model.to(DEVICE)
                print("best_model_" + name + "_classi_crop")
                evaluate(model, test_loader_crop)
                
        if YOLO_LIKE:
            backbone=["vgg19", "convnext_tiny", "Scratch_CNN"] 
            # Train
            for name in backbone:
                model= get_model("yolo",name)
                train(model, train_loader, val_loader, "yolo_" + name,5) 
                
                # Evaluation
                model.load_state_dict(torch.load("yolo_" + name + ".pth"))
                model.to(DEVICE)
                print("yolo_" + name)
                evaluate(model, test_loader)
                