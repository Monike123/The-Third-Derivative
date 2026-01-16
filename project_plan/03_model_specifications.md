# Model Specifications & Pretrained Models

## Complete Guide for All Detection Models

---

## 1. Model Overview

| Model | Purpose | Input | Output | Source |
|-------|---------|-------|--------|--------|
| Visual Detector | Image deepfake detection | 224x224 RGB | [real_prob, fake_prob] | Custom trained |
| Forensic Classifier | Forensic feature analysis | 12 features | [real_prob, fake_prob] | Custom trained |
| Audio Detector | Voice synthesis detection | Mel spectrogram | [real_prob, fake_prob] | Custom trained |
| Temporal Analyzer | Video frame consistency | 16 frames | [real_prob, fake_prob] | Custom trained |
| Face Detector | Face localization | Any size RGB | Bounding boxes | MTCNN (pretrained) |

---

## 2. Pretrained Models from Hugging Face

### 2.1 Image Deepfake Detectors

| Model | HuggingFace ID | Accuracy | Notes |
|-------|----------------|----------|-------|
| **Deepfake-ViT** | `dima806/deepfake_vs_real_image_detection` | ~95% | ViT-based, good for faces |
| **DFDC Winner** | `microsoft/resnet-50` | ~90% | Fine-tune on deepfake data |
| **SigLIP Detector** | `google/siglip-base-patch16-224` | ~92% | Multi-modal capable |

### 2.2 Usage Example

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Load pretrained deepfake detector
model_id = "dima806/deepfake_vs_real_image_detection"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

def detect_deepfake(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    return {
        "real_probability": probs[0][0].item(),
        "fake_probability": probs[0][1].item()
    }
```

---

## 3. Visual Detector Specification

### Architecture: EfficientNet-B4

```
Input: (batch, 3, 224, 224) - RGB normalized
├── EfficientNet-B4 Backbone (pretrained ImageNet)
│   └── Output: (batch, 1792) features
├── Dropout(0.3)
├── Linear(1792, 512) + ReLU
├── Dropout(0.2)
└── Linear(512, 2)
Output: (batch, 2) - [real_logit, fake_logit]
```

### Preprocessing

```python
# Normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return image[np.newaxis, ...]  # Add batch dim
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Scheduler | Cosine Annealing |
| Epochs | 50 |
| Batch Size | 32 |
| Label Smoothing | 0.1 |

---

## 4. Forensic Classifier Specification

### Architecture: MLP

```
Input: (batch, 12) - normalized features
├── Linear(12, 64) + BatchNorm + ReLU + Dropout(0.3)
├── Linear(64, 32) + BatchNorm + ReLU + Dropout(0.3)
└── Linear(32, 2)
Output: (batch, 2) - [real_logit, fake_logit]
```

### Input Features (12 total)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | high_freq_ratio | FFT high frequency energy ratio | 0-1 |
| 2 | spectral_entropy | Entropy of frequency spectrum | 0-10 |
| 3 | magnitude_mean | Mean FFT magnitude | 0-1000 |
| 4 | magnitude_std | Std dev of FFT magnitude | 0-500 |
| 5 | noise_variance | Overall noise variance | 0-100 |
| 6 | noise_mean | Mean absolute noise | 0-50 |
| 7 | noise_var_r | Red channel noise variance | 0-100 |
| 8 | noise_var_g | Green channel noise variance | 0-100 |
| 9 | noise_var_b | Blue channel noise variance | 0-100 |
| 10 | noise_var_ratio | Max/min channel noise ratio | 1-10 |
| 11 | sharpness | Laplacian variance (blur measure) | 0-5000 |
| 12 | blocking_score | JPEG block artifact measure | 0-20 |

### Feature Extraction Code

```python
from scipy import fftpack
from scipy.stats import entropy
import cv2
import numpy as np

def extract_forensic_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # FFT features
    f_transform = fftpack.fft2(gray)
    f_shift = fftpack.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]
    high_freq_ratio = 1 - np.sum(center) / np.sum(magnitude)
    
    mag_norm = magnitude.flatten() / np.sum(magnitude)
    spectral_ent = entropy(mag_norm + 1e-10)
    
    # Noise features
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    noise = image.astype(float) - denoised.astype(float)
    
    # Quality features
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'high_freq_ratio': high_freq_ratio,
        'spectral_entropy': spectral_ent,
        'magnitude_mean': np.mean(magnitude),
        'magnitude_std': np.std(magnitude),
        'noise_variance': np.var(noise),
        'noise_mean': np.mean(np.abs(noise)),
        'noise_var_r': np.var(noise[:,:,0]),
        'noise_var_g': np.var(noise[:,:,1]),
        'noise_var_b': np.var(noise[:,:,2]),
        'noise_var_ratio': max(np.var(noise[:,:,i]) for i in range(3)) / 
                          (min(np.var(noise[:,:,i]) for i in range(3)) + 1e-10),
        'sharpness': sharpness,
        'blocking_score': 0  # Compute if needed
    }
```

---

## 5. Audio Detector Specification

### Architecture: ResNet18 adapted for spectrograms

```
Input: (batch, 1, 80, 251) - Mel spectrogram
├── ResNet18 (in_channels=1)
│   └── Output: (batch, 512) features
├── Dropout(0.3)
├── Linear(512, 128) + ReLU
└── Linear(128, 2)
Output: (batch, 2) - [real_logit, synthetic_logit]
```

### Audio Preprocessing

```python
import librosa

def preprocess_audio(audio_path, target_sr=16000, duration=4.0):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Pad or trim
    n_samples = int(target_sr * duration)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=target_sr,
        n_fft=1024, hop_length=256, n_mels=80
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db[np.newaxis, np.newaxis, ...]  # (1, 1, 80, T)
```

---

## 6. Temporal Analyzer Specification

### Architecture: 3D CNN

```
Input: (batch, 16, 3, 224, 224) - 16 RGB frames
├── Permute to (batch, 3, 16, 224, 224)
├── Conv3D(3, 64, k=3x7x7, s=1x2x2) + BN + ReLU
├── MaxPool3D(k=1x3x3, s=1x2x2)
├── Conv3D(64, 128, k=3x3x3) + BN + ReLU
├── MaxPool3D(k=2x2x2)
├── Conv3D(128, 256, k=3x3x3) + BN + ReLU
├── MaxPool3D(k=2x2x2)
├── Conv3D(256, 512, k=3x3x3) + BN + ReLU
├── AdaptiveAvgPool3D(1, 1, 1)
├── Flatten + Dropout(0.5)
├── Linear(512, 128) + ReLU
└── Linear(128, 2)
Output: (batch, 2) - [consistent_logit, inconsistent_logit]
```

### Frame Sampling

```python
def sample_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.array(frames[:num_frames])
```

---

## 7. Face Detector (MTCNN)

### Usage

```python
from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

def detect_faces(image):
    """
    Args:
        image: RGB numpy array (H, W, 3)
    Returns:
        List of dicts with 'bbox' and 'confidence'
    """
    boxes, probs = mtcnn.detect(image)
    
    if boxes is None:
        return []
    
    faces = []
    for box, prob in zip(boxes, probs):
        if prob > 0.9:  # Confidence threshold
            x1, y1, x2, y2 = box.astype(int)
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(prob)
            })
    
    return faces
```

---

## 8. Model Export Formats

### ONNX Export

```python
import torch

def export_to_onnx(model, input_shape, save_path):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
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

# Export examples
export_to_onnx(visual_model, (1, 3, 224, 224), "visual_detector.onnx")
export_to_onnx(forensic_model, (1, 12), "forensic_classifier.onnx")
export_to_onnx(audio_model, (1, 1, 80, 251), "audio_detector.onnx")
```

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np

def load_onnx_model(model_path, use_gpu=True):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, providers=providers)

def run_inference(session, input_data):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]
```

---

## 9. Model Performance Benchmarks

### Expected Accuracy

| Model | Dataset | Accuracy | AUC |
|-------|---------|----------|-----|
| Visual Detector | FaceForensics++ c23 | 88-92% | 0.94 |
| Visual Detector | Celeb-DF v2 | 80-85% | 0.88 |
| Forensic Classifier | Mixed dataset | 75-80% | 0.82 |
| Audio Detector | ASVspoof | 85-90% | 0.92 |

### Inference Speed (GPU: RTX 3080)

| Model | Batch Size | Time (ms) |
|-------|------------|-----------|
| Visual Detector | 1 | 15-20 |
| Forensic Classifier | 1 | 1-2 |
| Audio Detector | 1 | 25-30 |
| MTCNN Face Detection | 1 | 50-100 |

---

## 10. Model Files Checklist

After training, you should have these files in `backend/ml_models/`:

```
ml_models/
├── visual_detector.onnx        # ~50-100 MB
├── visual_detector.pt          # ~50-100 MB (TorchScript backup)
├── forensic_classifier.onnx    # ~1 MB
├── forensic_scaler.pkl         # ~1 KB (scikit-learn scaler)
├── audio_detector.onnx         # ~40-80 MB
└── temporal_analyzer.onnx      # ~80-150 MB
```
