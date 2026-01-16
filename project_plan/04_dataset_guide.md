# Dataset Guide

## Complete Guide for Deepfake Detection Datasets

---

## 1. Primary Datasets Overview

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| FaceForensics++ | 1,000 original + 4,000 fake videos | 4 manipulation methods | Request access |
| Celeb-DF v2 | 590 real + 5,639 fake | Celebrity deepfakes | GitHub |
| DFDC | 100,000+ videos | Facebook challenge | Kaggle |
| Real vs Fake Faces | 140,000 images | StyleGAN generated | Kaggle |
| DeeperForensics-1.0 | 60,000 videos | Real-world perturbations | Request access |

---

## 2. FaceForensics++

### Description
Most widely used academic benchmark. Contains videos manipulated with 4 methods at 3 compression levels.

### Manipulation Methods
- **Deepfakes**: Face swap using autoencoder
- **Face2Face**: Expression transfer
- **FaceSwap**: Graphics-based face swap
- **NeuralTextures**: Neural rendering manipulation

### Compression Levels
- **c0**: Raw (no compression)
- **c23**: High quality (YouTube standard)
- **c40**: Low quality (heavy compression)

### Download

```python
# 1. Request access at https://github.com/ondyari/FaceForensics
# 2. After approval, use provided download script

# Download script usage
python download-FaceForensics.py \
    /path/to/download \
    -d Deepfakes Face2Face FaceSwap NeuralTextures \
    -c c23 \
    -t videos
```

### Directory Structure
```
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── 000.mp4
│               └── ...
├── manipulated_sequences/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
```

---

## 3. Celeb-DF v2

### Description
High-quality celebrity deepfakes with improved visual quality.

### Download

```bash
# Clone repository
git clone https://github.com/yuezunli/celeb-deepfakeforensics.git

# Download links are in the README
# Requires gdrive or similar tool for Google Drive download
```

### Structure
```
Celeb-DF-v2/
├── Celeb-real/       # 590 real videos
├── Celeb-synthesis/  # 5,639 fake videos
├── YouTube-real/     # Additional real videos
└── List_of_testing_videos.txt
```

---

## 4. DFDC (Deepfake Detection Challenge)

### Description
Largest public deepfake dataset from Facebook AI.

### Download from Kaggle

```python
# Install kaggle CLI
pip install kaggle

# Setup credentials
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
kaggle competitions download -c deepfake-detection-challenge

# Download sample dataset (smaller, for testing)
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
```

### Structure
```
dfdc_train_part_0/
├── metadata.json
├── aaa.mp4
├── aab.mp4
└── ...
```

### Metadata Format
```json
{
  "aaa.mp4": {
    "label": "FAKE",
    "split": "train",
    "original": "bbb.mp4"
  }
}
```

---

## 5. Real vs Fake Faces (Images)

### Description
140,000 images for training face image classifiers. Mix of real faces and StyleGAN-generated faces.

### Download

```python
# Kaggle download
kaggle datasets download -d ciplab/real-and-fake-face-detection

# Unzip
unzip real-and-fake-face-detection.zip -d datasets/faces/
```

### Structure
```
real_and_fake_face/
├── training_real/
│   ├── real_00001.jpg
│   └── ...
├── training_fake/
│   ├── easy/
│   ├── mid/
│   └── hard/
```

---

## 6. Audio Datasets

### ASVspoof 2019

For voice synthesis and spoofing detection.

```bash
# Download from https://www.asvspoof.org/
# Requires registration

# Structure
ASVspoof2019/
├── LA/
│   ├── ASVspoof2019_LA_train/
│   ├── ASVspoof2019_LA_dev/
│   └── ASVspoof2019_LA_eval/
```

### VoxCeleb2

For speaker verification and deepfake audio detection.

```python
# Download using voxceleb_trainer
pip install voxceleb_trainer
python download_voxceleb.py
```

---

## 7. Data Preprocessing Pipeline

### 7.1 Video to Frames

```python
import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames from video at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

# Batch process
def process_dataset(video_dir, output_dir, fps=1):
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    
    for video in tqdm(videos, desc="Processing videos"):
        video_path = os.path.join(video_dir, video)
        frame_dir = os.path.join(output_dir, video.replace('.mp4', ''))
        extract_frames(video_path, frame_dir, fps)
```

### 7.2 Face Extraction

```python
from facenet_pytorch import MTCNN
import torch
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device, min_face_size=100)

def extract_face(image_path, output_path, margin=0.2):
    """Extract and save face from image."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes, probs = mtcnn.detect(image_rgb)
    
    if boxes is None:
        return False
    
    # Take highest confidence face
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box
    
    # Add margin
    w, h = x2 - x1, y2 - y1
    margin_w, margin_h = int(w * margin), int(h * margin)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(image.shape[1], x2 + margin_w)
    y2 = min(image.shape[0], y2 + margin_h)
    
    face = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, face)
    return True
```

### 7.3 Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training augmentations
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Validation (no augmentation except resize and normalize)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 8. Dataset Splitting

```python
from sklearn.model_selection import train_test_split
import glob

def prepare_splits(data_dir, test_size=0.2, val_size=0.1):
    """Prepare train/val/test splits."""
    
    # Get all image paths
    real_images = glob.glob(f"{data_dir}/real/**/*.jpg", recursive=True)
    fake_images = glob.glob(f"{data_dir}/fake/**/*.jpg", recursive=True)
    
    all_images = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    
    # First split: train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_images, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Second split: train / val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=42,
        stratify=y_trainval
    )
    
    print(f"Train: {len(X_train)} images")
    print(f"Val: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
```

---

## 9. Dataset Class

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake detection."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label

# Usage
train_dataset = DeepfakeDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 10. Quick Start: Download Sample Data

For quick testing, use this smaller dataset:

```python
# In Google Colab
!pip install gdown

# Download 140k Real and Fake Faces (smaller subset)
!gdown --id 1-1bL8YhDrGGXy6k

# Or use this command for Kaggle
!pip install kaggle
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d ciplab/real-and-fake-face-detection -p ./data
!unzip ./data/real-and-fake-face-detection.zip -d ./data/faces
```

---

## 11. Data Quality Checklist

Before training, verify:

- [ ] All images can be loaded without errors
- [ ] Face detection works on majority of images
- [ ] Labels are balanced or properly weighted
- [ ] Train/val/test splits don't overlap
- [ ] No data leakage (same identity in train and test)
- [ ] Compression levels match intended use case
