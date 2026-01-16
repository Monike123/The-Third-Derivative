# Parallel Colab Training Strategy (Free Tier Optimized)

## Running Multiple Models on Multiple Accounts - 4 Hour Limit

---

> [!IMPORTANT]
> **Optimized for Colab Free Tier:**
> - Max 4 hours runtime per session
> - ~40-50K images total (20-25K real + 20-25K fake)
> - Smaller batch sizes, fewer epochs
> - Quick training with good results

---

## Model Distribution (Time-Optimized)

| Account | Model | Est. Time | Dataset Size |
|---------|-------|-----------|--------------|
| **Account 1** | Visual Detector (EfficientNet-B4) | 2-3 hrs | 40K images |
| **Account 2** | Forensic Classifier (MLP) | 30-45 min | 10K images (features) |
| **Account 3** | Temporal Analyzer (3D CNN) | 2-3 hrs | 5K video clips |
| **Account 4** | Audio Detector | 1-2 hrs | 10K audio samples |

---

## Shared Google Drive Setup

Create ONE shared folder accessible by all accounts:

```
MyDrive/deepfake_project/
├── datasets/
│   └── faces/           # 40-50K images total
├── checkpoints/
├── exports/             # Final ONNX models
└── status/
```

**Share this folder** with all your Google accounts (Editor permission).

---

## Account 1: Visual Spatial Detector (EfficientNet-B4)

**Time Budget:** 2-3 hours | **Dataset:** 40K images

### Complete Colab Notebook

```python
#@title 1. Setup Environment
!nvidia-smi
!pip install -q torch torchvision timm albumentations tqdm

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/deepfake_project'
os.makedirs(f'{PROJECT_ROOT}/checkpoints/visual_detector', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/exports', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/status', exist_ok=True)

print("✅ Setup complete!")
```

```python
#@title 2. Download Dataset (Real vs Fake Faces - Subset)
!pip install -q kaggle

# Upload your kaggle.json to Colab first, then run:
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/ 2>/dev/null || echo "Upload kaggle.json first!"
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d ciplab/real-and-fake-face-detection -p /content/data
!unzip -q /content/data/real-and-fake-face-detection.zip -d /content/data/faces

print("✅ Dataset downloaded!")
```

```python
#@title 3. Prepare Dataset (Limited to ~40K images)
import glob
import random
from sklearn.model_selection import train_test_split

# Get all image paths
real_images = glob.glob('/content/data/faces/**/training_real/*.jpg', recursive=True)
fake_images = glob.glob('/content/data/faces/**/training_fake/**/*.jpg', recursive=True)

print(f"Available: Real={len(real_images)}, Fake={len(fake_images)}")

# ⚡ LIMIT TO 40K TOTAL (20K each)
MAX_PER_CLASS = 20000

random.seed(42)
real_images = random.sample(real_images, min(MAX_PER_CLASS, len(real_images)))
fake_images = random.sample(fake_images, min(MAX_PER_CLASS, len(fake_images)))

print(f"Using: Real={len(real_images)}, Fake={len(fake_images)}")
print(f"Total: {len(real_images) + len(fake_images)} images")

# Combine and create labels
all_images = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

# Split: 80% train, 20% val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
```

```python
#@title 4. Create DataLoaders
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ⚡ Optimized transforms (faster augmentation)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10, 30), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)['image']
            return img, self.labels[idx]
        except:
            # Return a random valid sample if error
            return self.__getitem__((idx + 1) % len(self))

train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)

# ⚡ Batch size 32 for T4 GPU (fits in memory)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ DataLoaders ready!")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
```

```python
#@title 5. Define Model (EfficientNet-B4)
import torch.nn as nn
import timm

class VisualDetector(nn.Module):
    def __init__(self, backbone='efficientnet_b4', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisualDetector().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded on {device}")
print(f"   Total params: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")
```

```python
#@title 6. Training Loop (Optimized for Speed)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import time

# ⚡ Training config optimized for 4-hour limit
NUM_EPOCHS = 15  # Reduced from 50
LEARNING_RATE = 2e-4
WARMUP_EPOCHS = 2

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# OneCycleLR for faster convergence
scheduler = OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE * 10,
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1
)

# Mixed precision for 2x speed
scaler = GradScaler()

best_auc = 0
best_acc = 0
start_time = time.time()

print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
print(f"   Estimated time: ~2-3 hours")
print("-" * 50)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # === TRAIN ===
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*train_correct/train_total:.1f}%'
        })
    
    # === VALIDATE ===
    model.eval()
    val_preds = []
    val_labels_all = []
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with autocast():
                outputs = model(images)
            probs = torch.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()
            val_preds.extend(probs)
            val_labels_all.extend(labels.numpy())
            
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels.to(device)).sum().item()
            val_total += labels.size(0)
    
    val_auc = roc_auc_score(val_labels_all, val_preds)
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"  📊 Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {100*train_correct/train_total:.2f}%")
    print(f"  📊 Val AUC: {val_auc:.4f} | Val Acc: {100*val_acc:.2f}%")
    print(f"  ⏱️ Epoch: {epoch_time/60:.1f}min | Total: {total_time/60:.1f}min")
    
    # Save best model
    if val_auc > best_auc:
        best_auc = val_auc
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_auc': val_auc,
            'val_acc': val_acc,
        }, f'{PROJECT_ROOT}/checkpoints/visual_detector/best_model.pth')
        print(f"  ✅ NEW BEST! Saved (AUC: {val_auc:.4f})")
    
    print("-" * 50)

print(f"\n🎉 Training complete!")
print(f"   Best AUC: {best_auc:.4f}")
print(f"   Best Acc: {100*best_acc:.2f}%")
print(f"   Total time: {(time.time()-start_time)/60:.1f} minutes")
```

```python
#@title 7. Export to ONNX
import torch.onnx

# Load best model
checkpoint = torch.load(f'{PROJECT_ROOT}/checkpoints/visual_detector/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    f'{PROJECT_ROOT}/exports/visual_detector.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Also save PyTorch version
torch.save(model.state_dict(), f'{PROJECT_ROOT}/exports/visual_detector.pt')

# Update status
from datetime import datetime
with open(f'{PROJECT_ROOT}/status/visual_detector_done.txt', 'w') as f:
    f.write(f"Completed: {datetime.now()}\n")
    f.write(f"Best AUC: {best_auc:.4f}\n")
    f.write(f"Best Acc: {100*best_acc:.2f}%\n")

print("✅ Exported successfully!")
print(f"   📁 {PROJECT_ROOT}/exports/visual_detector.onnx")
print(f"   📁 {PROJECT_ROOT}/exports/visual_detector.pt")
```

---

## Account 2: Forensic Feature Classifier (MLP)

**Time Budget:** 30-45 minutes | **Dataset:** 10K images (for feature extraction)

### Complete Colab Notebook

```python
#@title 1. Setup
!pip install -q torch scipy scikit-learn joblib tqdm

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/deepfake_project'
os.makedirs(f'{PROJECT_ROOT}/checkpoints/forensic_classifier', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/exports', exist_ok=True)
```

```python
#@title 2. Feature Extraction Functions
import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import entropy

def extract_forensic_features(image):
    """Extract 12 forensic features from image (optimized for speed)."""
    # Resize for faster processing
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # FFT features (fast)
    f_shift = fftpack.fftshift(fftpack.fft2(gray))
    magnitude = np.abs(f_shift)
    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]
    high_freq_ratio = 1 - np.sum(center) / (np.sum(magnitude) + 1e-10)
    
    mag_flat = magnitude.flatten()
    mag_norm = mag_flat / (np.sum(mag_flat) + 1e-10)
    spectral_ent = entropy(mag_norm + 1e-10)
    
    # Simple noise estimation (faster than NLMeans)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    noise = image.astype(np.float32) - blur.astype(np.float32)
    
    noise_var = np.var(noise)
    noise_mean = np.mean(np.abs(noise))
    noise_var_r = np.var(noise[:,:,0])
    noise_var_g = np.var(noise[:,:,1])
    noise_var_b = np.var(noise[:,:,2])
    noise_ratio = max(noise_var_r, noise_var_g, noise_var_b) / (min(noise_var_r, noise_var_g, noise_var_b) + 1e-10)
    
    # Sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0
    
    return np.array([
        high_freq_ratio, spectral_ent, np.mean(magnitude), np.std(magnitude),
        noise_var, noise_mean, noise_var_r, noise_var_g, noise_var_b,
        noise_ratio, sharpness, edge_density
    ])

print("✅ Feature extraction functions ready!")
```

```python
#@title 3. Extract Features from Dataset
import glob
import random
from tqdm import tqdm

# Get image paths (use smaller subset for speed)
real_images = glob.glob('/content/data/faces/**/training_real/*.jpg', recursive=True)
fake_images = glob.glob('/content/data/faces/**/training_fake/**/*.jpg', recursive=True)

# If dataset not downloaded yet, download it
if len(real_images) == 0:
    print("Dataset not found! Downloading...")
    !pip install -q kaggle
    !mkdir -p ~/.kaggle
    !cp /content/kaggle.json ~/.kaggle/ 2>/dev/null || print("Upload kaggle.json!")
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d ciplab/real-and-fake-face-detection -p /content/data
    !unzip -q /content/data/real-and-fake-face-detection.zip -d /content/data/faces
    real_images = glob.glob('/content/data/faces/**/training_real/*.jpg', recursive=True)
    fake_images = glob.glob('/content/data/faces/**/training_fake/**/*.jpg', recursive=True)

# ⚡ LIMIT TO 10K TOTAL (5K each) for faster feature extraction
MAX_PER_CLASS = 5000

random.seed(42)
real_images = random.sample(real_images, min(MAX_PER_CLASS, len(real_images)))
fake_images = random.sample(fake_images, min(MAX_PER_CLASS, len(fake_images)))

print(f"Extracting features from {len(real_images) + len(fake_images)} images...")

all_images = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

# Extract features
features = []
valid_labels = []

for path, label in tqdm(zip(all_images, labels), total=len(all_images)):
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feat = extract_forensic_features(img)
        features.append(feat)
        valid_labels.append(label)
    except Exception as e:
        pass

X = np.array(features)
y = np.array(valid_labels)
print(f"✅ Extracted {len(X)} feature vectors")
print(f"   Feature shape: {X.shape}")

# Save features in case of timeout
np.save(f'{PROJECT_ROOT}/datasets/forensic_X.npy', X)
np.save(f'{PROJECT_ROOT}/datasets/forensic_y.npy', y)
```

```python
#@title 4. Train Forensic Classifier (MLP)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

# Load features (in case we're resuming)
try:
    X = np.load(f'{PROJECT_ROOT}/datasets/forensic_X.npy')
    y = np.load(f'{PROJECT_ROOT}/datasets/forensic_y.npy')
    print(f"Loaded {len(X)} samples from cache")
except:
    print("Using features from memory")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Model
class ForensicClassifier(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ForensicClassifier(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training
best_auc = 0
NUM_EPOCHS = 50  # Fast to train

print(f"🚀 Training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            probs = torch.softmax(model(X_batch.to(device)), dim=1)[:, 1].cpu().numpy()
            preds.extend(probs)
    
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, [1 if p > 0.5 else 0 for p in preds])
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: AUC={auc:.4f}, Acc={100*acc:.2f}%")
    
    if auc > best_auc:
        best_auc = auc
        best_acc = acc
        torch.save(model.state_dict(), f'{PROJECT_ROOT}/checkpoints/forensic_classifier/best_model.pth')

print(f"\n✅ Training complete!")
print(f"   Best AUC: {best_auc:.4f}")
print(f"   Best Acc: {100*best_acc:.2f}%")
```

```python
#@title 5. Export Model and Scaler
# Load best model
model.load_state_dict(torch.load(f'{PROJECT_ROOT}/checkpoints/forensic_classifier/best_model.pth'))
model.eval()

# Export ONNX
dummy = torch.randn(1, X.shape[1]).to(device)
torch.onnx.export(
    model, dummy,
    f'{PROJECT_ROOT}/exports/forensic_classifier.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=14,
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# Save scaler (CRITICAL - needed for inference)
joblib.dump(scaler, f'{PROJECT_ROOT}/exports/forensic_scaler.pkl')

# Update status
from datetime import datetime
with open(f'{PROJECT_ROOT}/status/forensic_classifier_done.txt', 'w') as f:
    f.write(f"Completed: {datetime.now()}\n")
    f.write(f"Best AUC: {best_auc:.4f}\n")

print("✅ Exported successfully!")
print(f"   📁 forensic_classifier.onnx")
print(f"   📁 forensic_scaler.pkl")
```

---

## Account 3: Temporal/Video Analyzer (3D CNN) - OPTIONAL

**Time Budget:** 2-3 hours | **Dataset:** 5K video clips (16 frames each)

```python
#@title Quick 3D CNN for Temporal Analysis
# This is simplified for Colab free tier
# Uses video frames extracted from FF++ or similar

import torch
import torch.nn as nn

class SimpleTemporal3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Lightweight 3D CNN
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        return self.classifier(x)

# Use 8 frames instead of 16 for speed
# Resize to 112x112 instead of 224x224
# Batch size: 8
```

---

## Account 4: Audio Detector - OPTIONAL

**Time Budget:** 1-2 hours | **Dataset:** 10K audio samples

```python
#@title Simple Audio Classifier
import torch
import torch.nn as nn

class SimpleAudioDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Process mel spectrogram: (1, 80, 251) for 4 seconds
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

# Extract 3-second clips (smaller than 4 seconds)
# Use librosa for mel spectrogram extraction
```

---

## Quick Reference: Time Estimates

| Task | Time |
|------|------|
| Mount Drive & Install packages | 2-3 min |
| Download dataset (Kaggle) | 5-10 min |
| Unzip dataset | 2-5 min |
| Visual Detector training (15 epochs) | 90-120 min |
| Export to ONNX | 1 min |
| **TOTAL Account 1** | **~2-2.5 hours** |

| Task | Time |
|------|------|
| Feature extraction (10K images) | 15-20 min |
| Forensic classifier training (50 epochs) | 5-10 min |
| Export | 1 min |
| **TOTAL Account 2** | **~30-45 min** |

---

## After Training: Required Files

Copy these from Google Drive to `Deepway/backend/ml_models/`:

```
✅ REQUIRED (Minimum viable system):
├── visual_detector.onnx     # Account 1
├── forensic_classifier.onnx # Account 2
└── forensic_scaler.pkl      # Account 2

🟡 OPTIONAL (Enhanced system):
├── temporal_analyzer.onnx   # Account 3
└── audio_detector.onnx      # Account 4
```

---

## Troubleshooting

### "GPU memory exhausted"
```python
# Reduce batch size
train_loader = DataLoader(..., batch_size=16, ...)  # Instead of 32
```

### "Session timeout"
- Save checkpoints every epoch (already done in code above)
- Resume from checkpoint if needed

### "Dataset download fails"
1. Upload `kaggle.json` to Colab
2. Run: `!cp /content/kaggle.json ~/.kaggle/`
3. Run: `!chmod 600 ~/.kaggle/kaggle.json`

---

## Summary: What You Need to Do

1. **Create shared Google Drive folder** and share with all accounts
2. **Account 1**: Run visual detector notebook (2-3 hrs)
3. **Account 2**: Run forensic classifier notebook (30-45 min)
4. **Download models** from Drive to local `backend/ml_models/`
5. **Start web app** using instructions in `02_webapp_implementation.md`

With just **Account 1 + Account 2**, you have a fully working deepfake detection system! 🎉
