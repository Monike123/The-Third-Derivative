# Web Application Implementation Guide

## Building the Deepfake Detection Web App

---

## 1. Project Structure

```
Deepway/
├── project_plan/               # Planning documents
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── config.py           # Configuration
│   │   ├── api/
│   │   │   ├── routes.py       # Route definitions
│   │   │   ├── image.py        # Image endpoints
│   │   │   └── video.py        # Video endpoints
│   │   ├── services/
│   │   │   ├── visual_detector.py
│   │   │   ├── forensic_analyzer.py
│   │   │   ├── fusion_engine.py
│   │   │   └── explainer.py
│   │   └── models/
│   │       └── response.py
│   ├── ml_models/              # Trained models (.onnx files)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/api.ts
│   │   └── App.tsx
│   └── package.json
└── tests/
```

---

## 2. Backend (FastAPI)

### requirements.txt

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
torch>=2.0.0
onnxruntime>=1.16.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
facenet-pytorch>=2.5.3
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
joblib>=1.3.0
librosa>=0.10.1
pydantic>=2.5.0
python-dotenv>=1.0.0
```

### main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import routes
from app.services import visual_detector, forensic_analyzer, fusion_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    visual_detector.load_model()
    forensic_analyzer.load_model()
    fusion_engine.initialize()
    yield

app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## 3. Visual Detector Service

```python
# backend/app/services/visual_detector.py
import numpy as np
import cv2
import onnxruntime as ort
from facenet_pytorch import MTCNN
import torch

class VisualDetectorService:
    def __init__(self):
        self.session = None
        self.mtcnn = None
        self._loaded = False
        
    def load_model(self):
        model_path = "ml_models/visual_detector.onnx"
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        self._loaded = True
    
    def preprocess(self, image):
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        return image.astype(np.float32)
    
    def detect_faces(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None:
            return []
        faces = []
        for box, prob in zip(boxes, probs):
            if prob > 0.9:
                x1, y1, x2, y2 = box.astype(int)
                faces.append({'bbox': [x1, y1, x2, y2], 'crop': image[y1:y2, x1:x2]})
        return faces
    
    def predict(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
        return {'fake_probability': float(probs[1]), 'real_probability': float(probs[0])}
    
    def analyze(self, image):
        faces = self.detect_faces(image)
        if not faces:
            prediction = self.predict(image)
            return {'overall_prediction': prediction, 'faces_detected': 0}
        
        face_preds = [self.predict(f['crop']) for f in faces]
        max_fake = max(p['fake_probability'] for p in face_preds)
        return {
            'overall_prediction': {'fake_probability': max_fake},
            'faces_detected': len(faces),
            'face_predictions': face_preds
        }

visual_detector = VisualDetectorService()
```

---

## 4. Forensic Analyzer Service

```python
# backend/app/services/forensic_analyzer.py
import numpy as np
import cv2
import onnxruntime as ort
from scipy import fftpack
from scipy.stats import entropy
import joblib

class ForensicAnalyzerService:
    def __init__(self):
        self.session = None
        self.scaler = None
        
    def load_model(self):
        self.session = ort.InferenceSession("ml_models/forensic_classifier.onnx")
        self.scaler = joblib.load("ml_models/forensic_scaler.pkl")
    
    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # FFT features
        f_shift = fftpack.fftshift(fftpack.fft2(gray))
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape
        low_freq = magnitude[h//4:3*h//4, w//4:3*w//4]
        high_freq_ratio = 1 - np.sum(low_freq) / np.sum(magnitude)
        spectral_ent = entropy(magnitude.flatten() / np.sum(magnitude))
        
        # Noise features
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        noise = image.astype(float) - denoised.astype(float)
        
        return [high_freq_ratio, spectral_ent, np.mean(magnitude), np.std(magnitude),
                np.var(noise), np.mean(np.abs(noise)), np.var(noise[:,:,0]),
                np.var(noise[:,:,1]), np.var(noise[:,:,2]), 
                max(np.var(noise[:,:,i]) for i in range(3)) / (min(np.var(noise[:,:,i]) for i in range(3)) + 1e-10),
                cv2.Laplacian(gray, cv2.CV_64F).var(), 0]  # blocking_score simplified
    
    def analyze(self, image):
        features = np.array([self.extract_features(image)])
        scaled = self.scaler.transform(features)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: scaled.astype(np.float32)})
        probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
        return {'prediction': {'fake_probability': float(probs[1])}, 'features': dict(zip(
            ['high_freq_ratio', 'spectral_entropy', 'magnitude_mean', 'magnitude_std',
             'noise_variance', 'noise_mean', 'noise_var_r', 'noise_var_g', 'noise_var_b',
             'noise_var_ratio', 'sharpness', 'blocking_score'], features[0]))}

forensic_analyzer = ForensicAnalyzerService()
```

---

## 5. Fusion Engine

```python
# backend/app/services/fusion_engine.py
import numpy as np

class FusionEngine:
    VISUAL_WEIGHT = 0.45
    FORENSIC_WEIGHT = 0.30
    TEMPORAL_WEIGHT = 0.15
    AUDIO_WEIGHT = 0.10
    
    def initialize(self): pass
    
    def fuse_image_signals(self, visual_result, forensic_result):
        signals = {}
        if visual_result.get('overall_prediction'):
            signals['visual'] = visual_result['overall_prediction']['fake_probability']
        if forensic_result.get('prediction'):
            signals['forensic'] = forensic_result['prediction']['fake_probability']
        
        weights = {'visual': self.VISUAL_WEIGHT, 'forensic': self.FORENSIC_WEIGHT}
        risk_score = sum(signals[k] * weights[k] for k in signals) / sum(weights[k] for k in signals) * 100
        
        classification = 'MANIPULATED' if risk_score >= 70 else 'SUSPICIOUS' if risk_score >= 40 else 'AUTHENTIC'
        confidence = 'HIGH' if abs(risk_score - 50) > 35 else 'MEDIUM' if abs(risk_score - 50) > 15 else 'LOW'
        
        return {'risk_score': risk_score, 'classification': classification, 'confidence': confidence}

fusion_engine = FusionEngine()
```

---

## 6. Image Analysis Endpoint

```python
# backend/app/api/image.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import io
import uuid
from datetime import datetime
from app.services.visual_detector import visual_detector
from app.services.forensic_analyzer import forensic_analyzer
from app.services.fusion_engine import fusion_engine

router = APIRouter()

@router.post("/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    
    visual_result = visual_detector.analyze(image_np)
    forensic_result = forensic_analyzer.analyze(image_np)
    fused_result = fusion_engine.fuse_image_signals(visual_result, forensic_result)
    
    return {
        'analysis_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'classification': fused_result['classification'],
        'confidence': fused_result['confidence'],
        'risk_score': fused_result['risk_score'],
        'signals': {
            'visual': {'score': visual_result['overall_prediction']['fake_probability']},
            'forensic': {'score': forensic_result['prediction']['fake_probability']}
        }
    }
```

---

## 7. Frontend (React + TypeScript)

### Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install axios react-router-dom react-dropzone framer-motion lucide-react recharts
```

### API Service (src/services/api.ts)

```typescript
import axios from 'axios';
const api = axios.create({ baseURL: 'http://localhost:8000/api/v1', timeout: 60000 });

export const analyzeImage = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/analyze/image/', formData);
  return response.data;
};
```

---

## 8. Running the Application

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## 9. Deployment with Docker

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./backend/ml_models:/app/ml_models"]
  frontend:
    build: ./frontend
    ports: ["3000:80"]
```
