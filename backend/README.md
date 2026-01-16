# Deepfake Detection API - Backend

## FastAPI Application for Deepfake Detection

### Quick Start
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### API Endpoints
- `GET /health` - Health check
- `POST /api/v1/analyze/image/` - Analyze image
- `POST /api/v1/analyze/video/` - Analyze video
- `POST /api/v1/analyze/audio/` - Analyze audio
