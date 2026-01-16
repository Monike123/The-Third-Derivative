# Google Colab Model Training Guide

## Complete Guide for Training Deepfake Detection Models

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Visual Detector Training](#3-visual-detector-training)
4. [Forensic Classifier Training](#4-forensic-classifier-training)
5. [Audio Detector Training](#5-audio-detector-training)
6. [Temporal Analyzer Training](#6-temporal-analyzer-training)
7. [Model Export](#7-model-export)
8. [Model Upload](#8-model-upload)

---

## 1. Environment Setup

### 1.1 Colab Configuration

> [!IMPORTANT]
> Use **Colab Pro** or **Colab Pro+** for training. Free tier has GPU limitations.

```python
# Check GPU availability
!nvidia-smi

# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install timm transformers datasets wandb
!pip install opencv-python-headless pillow
!pip install onnx onnxruntime
!pip install facenet-pytorch  # For face detection
!pip install librosa soundfile  # For audio processing
!pip install albumentations  # For augmentations
```

### 1.2 Google Drive Mount

```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directories
import os
PROJECT_ROOT = '/content/drive/MyDrive/deepfake_detection'
os.makedirs(f'{PROJECT_ROOT}/datasets', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/checkpoints', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/exports', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/logs', exist_ok=True)
```

### 1.3 Weights & Biases Setup (Optional)

```python
import wandb
wandb.login()
wandb.init(project="deepfake-detection", entity="your-username")
```

---

## 2. Dataset Preparation

### 2.1 Dataset Sources

| Dataset | Size | Description | Download |
|---------|------|-------------|----------|
| FaceForensics++ | ~1000 videos | 4 manipulation methods | [GitHub](https://github.com/ondyari/FaceForensics) |
| Celeb-DF v2 | ~6000 videos | High-quality celebrity deepfakes | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| DFDC | ~100K videos | Facebook challenge dataset | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| Real vs Fake Faces | ~140K images | StyleGAN generated faces | [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) |

### 2.2 Download Scripts

```python
# FaceForensics++ Download (requires access request)
# After getting access, use their download script

# For Kaggle datasets
!pip install kaggle
!mkdir -p ~/.kaggle
# Upload your kaggle.json API key
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download Real vs Fake Faces dataset
!kaggle datasets download -d ciplab/real-and-fake-face-detection -p {PROJECT_ROOT}/datasets
!unzip {PROJECT_ROOT}/datasets/real-and-fake-face-detection.zip -d {PROJECT_ROOT}/datasets/faces
```

### 2.3 Data Preprocessing Pipeline

```python
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize face detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection training."""
    
    def __init__(self, image_paths, labels, transform=None, face_crop=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.face_crop = face_crop
        
    def __len__(self):
        return len(self.image_paths)
    
    def extract_face(self, image):
        """Extract face region from image."""
        try:
            boxes, _ = mtcnn.detect(image)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                # Add margin
                margin = int((x2 - x1) * 0.2)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.shape[1], x2 + margin)
                y2 = min(image.shape[0], y2 + margin)
                return image[y1:y2, x1:x2]
        except:
            pass
        return image
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.face_crop:
            image = self.extract_face(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label

# Training augmentations
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Validation augmentations
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 2.4 Data Loading Utility

```python
import glob
from sklearn.model_selection import train_test_split

def prepare_dataset(data_dir):
    """Prepare train/val splits from directory structure."""
    
    real_images = glob.glob(f'{data_dir}/real/**/*.jpg', recursive=True)
    fake_images = glob.glob(f'{data_dir}/fake/**/*.jpg', recursive=True)
    
    # Also check for png
    real_images += glob.glob(f'{data_dir}/real/**/*.png', recursive=True)
    fake_images += glob.glob(f'{data_dir}/fake/**/*.png', recursive=True)
    
    all_images = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)  # 0=real, 1=fake
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print(f"Real/Fake ratio: {labels.count(0)}/{labels.count(1)}")
    
    return train_paths, val_paths, train_labels, val_labels

# Prepare datasets
train_paths, val_paths, train_labels, val_labels = prepare_dataset(
    f'{PROJECT_ROOT}/datasets/faces'
)

train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
```

---

## 3. Visual Detector Training

### 3.1 Model Architecture - EfficientNet B4

```python
import torch
import torch.nn as nn
import timm

class DeepfakeVisualDetector(nn.Module):
    """EfficientNet-B4 based deepfake detector."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4', 
            pretrained=pretrained, 
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features for fusion."""
        return self.backbone(x)

# Alternative: Vision Transformer
class DeepfakeViTDetector(nn.Module):
    """Vision Transformer based deepfake detector."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
```

### 3.2 Training Loop

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return avg_loss, auc, acc

# Initialize training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeVisualDetector(num_classes=2, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Training loop
best_auc = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Log to wandb
    wandb.log({
        'train_loss': train_loss,
        'train_auc': train_auc,
        'val_loss': val_loss,
        'val_auc': val_auc,
        'val_acc': val_acc,
        'lr': scheduler.get_last_lr()[0]
    })
    
    # Save best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }, f'{PROJECT_ROOT}/checkpoints/visual_detector_best.pth')
        print(f"Saved best model with AUC: {val_auc:.4f}")

print(f"\nTraining complete! Best AUC: {best_auc:.4f}")
```

---

## 4. Forensic Classifier Training

### 4.1 Feature Extraction

```python
import numpy as np
from scipy import fftpack
from scipy.stats import entropy

class ForensicFeatureExtractor:
    """Extract forensic features for deepfake detection."""
    
    def extract_fft_features(self, image):
        """Extract frequency domain features."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 2D FFT
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High frequency ratio
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        low_freq = magnitude[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        high_freq_ratio = 1 - (np.sum(low_freq) / np.sum(magnitude))
        
        # Spectral entropy
        magnitude_flat = magnitude.flatten()
        magnitude_norm = magnitude_flat / np.sum(magnitude_flat)
        spectral_entropy = entropy(magnitude_norm + 1e-10)
        
        return {
            'high_freq_ratio': high_freq_ratio,
            'spectral_entropy': spectral_entropy,
            'magnitude_mean': np.mean(magnitude),
            'magnitude_std': np.std(magnitude)
        }
    
    def extract_noise_features(self, image):
        """Extract noise residual features."""
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Noise residual
        noise = image.astype(np.float32) - denoised.astype(np.float32)
        
        # Noise statistics
        noise_var = np.var(noise)
        noise_mean = np.mean(np.abs(noise))
        
        # Per-channel variance
        noise_var_r = np.var(noise[:, :, 0])
        noise_var_g = np.var(noise[:, :, 1])
        noise_var_b = np.var(noise[:, :, 2])
        
        return {
            'noise_variance': noise_var,
            'noise_mean': noise_mean,
            'noise_var_r': noise_var_r,
            'noise_var_g': noise_var_g,
            'noise_var_b': noise_var_b,
            'noise_var_ratio': max(noise_var_r, noise_var_g, noise_var_b) / 
                               (min(noise_var_r, noise_var_g, noise_var_b) + 1e-10)
        }
    
    def extract_quality_features(self, image):
        """Extract image quality features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # JPEG artifact detection (blocking)
        h, w = gray.shape
        block_size = 8
        block_diffs = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                if j + block_size < w:
                    diff = np.abs(gray[i:i+block_size, j+block_size-1].astype(float) - 
                                  gray[i:i+block_size, j+block_size].astype(float))
                    block_diffs.append(np.mean(diff))
        
        blocking_score = np.mean(block_diffs) if block_diffs else 0
        
        return {
            'sharpness': laplacian_var,
            'blocking_score': blocking_score
        }
    
    def extract_all_features(self, image):
        """Extract all forensic features."""
        features = {}
        features.update(self.extract_fft_features(image))
        features.update(self.extract_noise_features(image))
        features.update(self.extract_quality_features(image))
        return features

# Feature extraction for dataset
extractor = ForensicFeatureExtractor()

def extract_features_for_dataset(image_paths, labels):
    """Extract features for all images."""
    all_features = []
    all_labels = []
    
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features = extractor.extract_all_features(image)
            all_features.append(list(features.values()))
            all_labels.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return np.array(all_features), np.array(all_labels)

# Extract features
print("Extracting forensic features...")
X_train, y_train = extract_features_for_dataset(train_paths[:5000], train_labels[:5000])
X_val, y_val = extract_features_for_dataset(val_paths, val_labels)

# Save features
np.save(f'{PROJECT_ROOT}/datasets/forensic_X_train.npy', X_train)
np.save(f'{PROJECT_ROOT}/datasets/forensic_y_train.npy', y_train)
np.save(f'{PROJECT_ROOT}/datasets/forensic_X_val.npy', X_val)
np.save(f'{PROJECT_ROOT}/datasets/forensic_y_val.npy', y_val)
```

### 4.2 Forensic Classifier Model

```python
class ForensicClassifier(nn.Module):
    """MLP classifier for forensic features."""
    
    def __init__(self, input_dim=12, hidden_dims=[64, 32], num_classes=2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Create tensors
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.LongTensor(y_val)

train_forensic_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_forensic_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

train_forensic_loader = DataLoader(train_forensic_dataset, batch_size=64, shuffle=True)
val_forensic_loader = DataLoader(val_forensic_dataset, batch_size=64, shuffle=False)

# Train forensic classifier
forensic_model = ForensicClassifier(input_dim=X_train.shape[1]).to(device)
forensic_criterion = nn.CrossEntropyLoss()
forensic_optimizer = optim.Adam(forensic_model.parameters(), lr=1e-3)

for epoch in range(30):
    forensic_model.train()
    for X_batch, y_batch in train_forensic_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        forensic_optimizer.zero_grad()
        outputs = forensic_model(X_batch)
        loss = forensic_criterion(outputs, y_batch)
        loss.backward()
        forensic_optimizer.step()
    
    # Validate
    forensic_model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_forensic_loader:
            X_batch = X_batch.to(device)
            outputs = forensic_model(X_batch)
            val_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Epoch {epoch+1}/30, Val AUC: {val_auc:.4f}")

# Save scaler and model
import joblib
joblib.dump(scaler, f'{PROJECT_ROOT}/exports/forensic_scaler.pkl')
torch.save(forensic_model.state_dict(), f'{PROJECT_ROOT}/checkpoints/forensic_classifier.pth')
```

---

## 5. Audio Detector Training

### 5.1 Audio Preprocessing

```python
import librosa
import soundfile as sf

class AudioProcessor:
    """Audio preprocessing for voice synthesis detection."""
    
    def __init__(self, sample_rate=16000, duration=4.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
    def load_audio(self, path):
        """Load and preprocess audio."""
        audio, sr = librosa.load(path, sr=self.sample_rate)
        
        # Pad or trim to fixed length
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        else:
            audio = audio[:self.n_samples]
            
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, audio):
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=40
        )
        return mfcc

class AudioDataset(Dataset):
    """Dataset for audio deepfake detection."""
    
    def __init__(self, audio_paths, labels, processor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio = self.processor.load_audio(self.audio_paths[idx])
        mel_spec = self.processor.extract_mel_spectrogram(audio)
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dim
        return mel_spec, self.labels[idx]
```

### 5.2 Audio Classifier Model

```python
class AudioDeepfakeDetector(nn.Module):
    """CNN-based audio deepfake detector using mel spectrograms."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use pretrained ResNet18 adapted for spectrograms
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=True,
            in_chans=1,  # Single channel for spectrogram
            num_classes=0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Note: For production, consider using RawNet2 or AASIST architectures
# which work directly on raw waveforms
```

---

## 6. Temporal Analyzer Training

### 6.1 Video Frame Dataset

```python
class VideoFrameDataset(Dataset):
    """Dataset for temporal inconsistency detection."""
    
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def sample_frames(self, video_path):
        """Sample evenly spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    augmented = self.transform(image=frame)
                    frame = augmented['image']
                frames.append(frame)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.num_frames])
    
    def __getitem__(self, idx):
        frames = self.sample_frames(self.video_paths[idx])
        return frames, self.labels[idx]
```

### 6.2 3D CNN Model

```python
class Temporal3DCNN(nn.Module):
    """3D CNN for temporal inconsistency detection."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 3D Convolutional backbone
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        return self.classifier(x)
```

---

## 7. Model Export

### 7.1 Export to ONNX

```python
def export_to_onnx(model, input_shape, save_path, model_name):
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}/{model_name}.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Exported {model_name} to ONNX")

# Export visual detector
export_to_onnx(model, (1, 3, 224, 224), f'{PROJECT_ROOT}/exports', 'visual_detector')

# Export forensic classifier
export_to_onnx(forensic_model, (1, X_train.shape[1]), f'{PROJECT_ROOT}/exports', 'forensic_classifier')
```

### 7.2 Export to TorchScript

```python
def export_to_torchscript(model, input_shape, save_path, model_name):
    """Export PyTorch model to TorchScript format."""
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(f"{save_path}/{model_name}.pt")
    print(f"Exported {model_name} to TorchScript")

# Export models
export_to_torchscript(model, (1, 3, 224, 224), f'{PROJECT_ROOT}/exports', 'visual_detector')
```

---

## 8. Model Upload

### 8.1 Upload to Google Drive

Models are automatically saved to Google Drive through the mounted path.

```python
# Verify exports
import os
export_dir = f'{PROJECT_ROOT}/exports'
print("Exported models:")
for f in os.listdir(export_dir):
    size = os.path.getsize(f'{export_dir}/{f}') / (1024 * 1024)
    print(f"  {f}: {size:.2f} MB")
```

### 8.2 Download for Local Use

After training, download the models from Google Drive:

1. Navigate to `MyDrive/deepfake_detection/exports/`
2. Download all `.onnx` and `.pt` files
3. Place in your local `backend/models/` directory

### 8.3 Upload to Hugging Face (Optional)

```python
from huggingface_hub import HfApi, upload_file

api = HfApi()

# Create a new repo
api.create_repo(repo_id="your-username/deepfake-detector", exist_ok=True)

# Upload models
upload_file(
    path_or_fileobj=f"{PROJECT_ROOT}/exports/visual_detector.onnx",
    path_in_repo="visual_detector.onnx",
    repo_id="your-username/deepfake-detector"
)
```

---

## Training Checklist

- [ ] Set up Colab environment with GPU
- [ ] Download and prepare datasets
- [ ] Train Visual Detector (EfficientNet-B4)
- [ ] Train Forensic Classifier (MLP)
- [ ] Train Audio Detector (if audio data available)
- [ ] Train Temporal Analyzer (if video data available)
- [ ] Export all models to ONNX
- [ ] Export models to TorchScript (backup)
- [ ] Download models to local machine
- [ ] Test inference locally

---

## Troubleshooting

### GPU Memory Issues
```python
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=16, ...)

# Use gradient checkpointing
model.backbone.set_grad_checkpointing(True)

# Clear cache
torch.cuda.empty_cache()
```

### Slow Training
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Colab Disconnections
- Save checkpoints frequently
- Use Colab Pro for longer sessions
- Consider using `!pip install colabcode` for persistent sessions
