# Agent Handoff Document

## Autonomous AI Agent Instructions for Deepfake Detection Project

---

> [!IMPORTANT]
> This document is designed to be consumed by an AI agent to autonomously continue work on this project. Follow instructions precisely and in order.

---

## 1. Project Context

**Project Name:** Deepfake Detection & Media Authenticity Analyzer

**Goal:** Build an AI/ML system that detects deepfake images and videos with high accuracy.

**Architecture:**
- **Training:** Google Colab (model training)
- **Web App:** Local/Cloud (FastAPI + React)
- **Models:** Combination of pretrained (HuggingFace) + custom trained (Colab)

**Current Status:** Planning complete. Ready for implementation.

---

## 2. Project Structure to Create

```
Deepway/
├── project_plan/           # ✅ Already created
│   ├── 00_project_overview.md
│   ├── 01_colab_training_guide.md
│   ├── 02_webapp_implementation.md
│   ├── 03_model_specifications.md
│   ├── 04_dataset_guide.md
│   ├── 05_api_specifications.md
│   └── 99_agent_handoff.md
│
├── training/               # 🔨 Create this
│   ├── notebooks/
│   │   ├── 01_setup_environment.ipynb
│   │   ├── 02_data_preprocessing.ipynb
│   │   ├── 03_visual_detector_training.ipynb
│   │   ├── 04_forensic_classifier_training.ipynb
│   │   ├── 05_audio_detector_training.ipynb
│   │   └── 06_model_export.ipynb
│   └── README.md
│
├── backend/                # 🔨 Create this
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py
│   │   │   ├── image.py
│   │   │   ├── video.py
│   │   │   └── audio.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── visual_detector.py
│   │   │   ├── forensic_analyzer.py
│   │   │   ├── audio_detector.py
│   │   │   ├── fusion_engine.py
│   │   │   └── explainer.py
│   │   └── models/
│   │       ├── __init__.py
│   │       └── response.py
│   ├── ml_models/          # Empty, models added after training
│   │   └── .gitkeep
│   ├── requirements.txt
│   └── README.md
│
├── frontend/               # 🔨 Create this
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── README.md
│
└── tests/                  # 🔨 Create this
    ├── test_visual_detector.py
    ├── test_api.py
    └── sample_data/
```

---

## 3. Implementation Order

Execute these tasks in order:

### Phase A: Backend Setup (Do First)

1. **Create backend directory structure**
2. **Create requirements.txt**
3. **Create config.py**
4. **Create main.py with FastAPI app**
5. **Create service files (visual_detector.py, etc.)**
6. **Create API route files**
7. **Create response models**

### Phase B: Frontend Setup

1. **Initialize Vite + React + TypeScript project**
2. **Install dependencies**
3. **Create API service**
4. **Create components**
5. **Create pages**
6. **Apply styling**

### Phase C: Training Notebooks

1. **Create Colab notebook templates**
2. **Include all code from 01_colab_training_guide.md**

---

## 4. Backend Implementation Tasks

### Task 4.1: Create requirements.txt

**Path:** `backend/requirements.txt`

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
aiofiles==23.2.1
torch>=2.0.0
torchvision>=0.15.0
onnxruntime>=1.16.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
facenet-pytorch>=2.5.3
albumentations>=1.3.1
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
joblib>=1.3.0
pydantic>=2.5.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
```

### Task 4.2: Create config.py

**Path:** `backend/app/config.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    MODELS_DIR: Path = Path(__file__).parent.parent / "ml_models"
    VISUAL_DETECTOR_PATH: str = "visual_detector.onnx"
    FORENSIC_CLASSIFIER_PATH: str = "forensic_classifier.onnx"
    FORENSIC_SCALER_PATH: str = "forensic_scaler.pkl"
    
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    SUPPORTED_IMAGE_FORMATS: list = [".jpg", ".jpeg", ".png", ".webp"]
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".avi", ".mov", ".webm"]
    
    USE_GPU: bool = True
    VISUAL_WEIGHT: float = 0.45
    FORENSIC_WEIGHT: float = 0.30
    TEMPORAL_WEIGHT: float = 0.15
    AUDIO_WEIGHT: float = 0.10
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Task 4.3: Create main.py

**Path:** `backend/app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import routes
from app.config import settings
from app.services.visual_detector import visual_detector
from app.services.forensic_analyzer import forensic_analyzer
from app.services.fusion_engine import fusion_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    visual_detector.load_model()
    forensic_analyzer.load_model()
    fusion_engine.initialize()
    logger.info("Models loaded!")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Deepfake Detection API",
    description="AI system for detecting deepfake media",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "visual_detector": visual_detector.is_loaded(),
            "forensic_analyzer": forensic_analyzer.is_loaded()
        }
    }
```

### Task 4.4: Create Visual Detector Service

**Path:** `backend/app/services/visual_detector.py`

Reference: See `02_webapp_implementation.md` Section 3.1

Key implementation:
- Load ONNX model
- Initialize MTCNN face detector
- Preprocess images (resize, normalize)
- Detect faces and analyze each
- Return probabilities

### Task 4.5: Create Forensic Analyzer Service

**Path:** `backend/app/services/forensic_analyzer.py`

Reference: See `02_webapp_implementation.md` Section 3.2

Key implementation:
- Extract FFT features
- Extract noise features
- Extract quality features
- Run through MLP classifier
- Return feature importances

### Task 4.6: Create Fusion Engine

**Path:** `backend/app/services/fusion_engine.py`

Reference: See `02_webapp_implementation.md` Section 6

Key implementation:
- Combine signals with weights
- Compute risk score (0-100)
- Determine classification (AUTHENTIC/SUSPICIOUS/MANIPULATED)
- Compute confidence level

### Task 4.7: Create API Routes

**Path:** `backend/app/api/routes.py`

```python
from fastapi import APIRouter
from app.api import image, video, audio

router = APIRouter()
router.include_router(image.router, prefix="/analyze/image", tags=["Image"])
router.include_router(video.router, prefix="/analyze/video", tags=["Video"])
router.include_router(audio.router, prefix="/analyze/audio", tags=["Audio"])
```

### Task 4.8: Create Image Endpoint

**Path:** `backend/app/api/image.py`

Reference: See `02_webapp_implementation.md` Section 4.2

---

## 5. Frontend Implementation Tasks

### Task 5.1: Initialize Project

```bash
cd Deepway
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install axios react-router-dom react-dropzone framer-motion lucide-react recharts @tanstack/react-query
```

### Task 5.2: Create File Structure

```
frontend/src/
├── components/
│   ├── FileUpload.tsx
│   ├── AnalysisResults.tsx
│   ├── ConfidenceGauge.tsx
│   ├── SignalBreakdown.tsx
│   └── Header.tsx
├── pages/
│   ├── HomePage.tsx
│   ├── ImageAnalysis.tsx
│   └── VideoAnalysis.tsx
├── services/
│   └── api.ts
├── App.tsx
├── main.tsx
└── index.css
```

### Task 5.3: Key Components to Build

1. **FileUpload.tsx** - Drag-and-drop file upload with react-dropzone
2. **AnalysisResults.tsx** - Display classification, risk score, signals
3. **ConfidenceGauge.tsx** - Visual gauge showing risk level
4. **SignalBreakdown.tsx** - Show visual/forensic/temporal signals

---

## 6. Model Files Required

After Colab training, these files must be placed in `backend/ml_models/`:

| File | Size | Required |
|------|------|----------|
| visual_detector.onnx | ~50MB | Yes |
| forensic_classifier.onnx | ~1MB | Yes |
| forensic_scaler.pkl | ~1KB | Yes |
| audio_detector.onnx | ~40MB | Optional |

---

## 7. Testing Checklist

After implementation, verify:

1. [ ] Backend starts without errors: `uvicorn app.main:app --reload`
2. [ ] Health endpoint works: `GET /health`
3. [ ] Image analysis works: `POST /api/v1/analyze/image/`
4. [ ] Frontend builds: `npm run build`
5. [ ] Frontend connects to backend
6. [ ] File upload works
7. [ ] Results display correctly

---

## 8. Commands Reference

### Backend

```bash
# Setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --port 8000

# Test
curl http://localhost:8000/health
```

### Frontend

```bash
# Setup
cd frontend
npm install

# Run
npm run dev

# Build
npm run build
```

---

## 9. Common Issues & Solutions

### Issue: ONNX model not found
```
Solution: Models are loaded from backend/ml_models/
Ensure models are trained and exported from Colab first
```

### Issue: CUDA not available
```python
# In config.py, set:
USE_GPU: bool = False
```

### Issue: CORS errors
```python
# Ensure frontend origin is in ALLOWED_ORIGINS
ALLOWED_ORIGINS: list = ["http://localhost:5173"]
```

---

## 10. Next Agent Actions

When continuing this project, the next AI agent should:

1. **Read all documents in `project_plan/`** to understand the full scope
2. **Check current implementation status** - list files in backend/frontend
3. **Continue from where left off** - implement missing components
4. **Test incrementally** - verify each component works before moving on
5. **Update this document** with progress

---

## 11. Reference Documents

| Document | Purpose |
|----------|---------|
| 00_project_overview.md | High-level architecture |
| 01_colab_training_guide.md | Complete training code |
| 02_webapp_implementation.md | Backend + Frontend code |
| 03_model_specifications.md | Model architectures |
| 04_dataset_guide.md | Dataset sources |
| 05_api_specifications.md | API documentation |

---

## 12. Success Criteria

Project is complete when:

- [ ] All backend services implemented
- [ ] All API endpoints functional
- [ ] Frontend UI complete
- [ ] At least visual_detector model trained
- [ ] End-to-end test passes (upload image → get result)
- [ ] Documentation updated
