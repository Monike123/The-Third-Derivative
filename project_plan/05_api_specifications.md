# API Specifications

## Complete REST API Documentation

---

## 1. Base URL

```
Development: http://localhost:8000/api/v1
Production:  https://your-domain.com/api/v1
```

---

## 2. Authentication

Currently no authentication required. For production, add JWT or API key authentication.

```python
# Future: Add to headers
headers = {
    "Authorization": "Bearer <token>",
    "X-API-Key": "<api-key>"
}
```

---

## 3. Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze/image/` | Analyze single image |
| POST | `/analyze/image/batch` | Analyze multiple images |
| POST | `/analyze/video/` | Analyze video |
| POST | `/analyze/audio/` | Analyze audio |
| GET | `/results/{analysis_id}` | Get cached result |

---

## 4. Health Check

### Request
```http
GET /api/v1/health
```

### Response
```json
{
  "status": "healthy",
  "models_loaded": {
    "visual_detector": true,
    "forensic_analyzer": true,
    "audio_detector": true
  },
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

---

## 5. Image Analysis

### 5.1 Single Image

#### Request
```http
POST /api/v1/analyze/image/
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Image file (JPEG, PNG, WebP) |

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/image/" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

#### Response
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T12:00:00Z",
  "media_type": "image",
  "filename": "test_image.jpg",
  "classification": "MANIPULATED",
  "confidence": "HIGH",
  "risk_score": 78.5,
  "signals": {
    "visual": {
      "score": 0.82,
      "weight": 0.45,
      "faces_detected": 1,
      "analysis_type": "face_level"
    },
    "forensic": {
      "score": 0.71,
      "weight": 0.30,
      "features": {
        "high_freq_ratio": 0.23,
        "spectral_entropy": 4.8,
        "noise_variance": 45.2,
        "sharpness": 1250.0
      }
    }
  },
  "face_detections": [
    {
      "face_id": 0,
      "bbox": [120, 80, 320, 280],
      "detection_confidence": 0.98,
      "fake_probability": 0.82
    }
  ],
  "explanation": {
    "summary": "This media shows strong signs of manipulation with a risk score of 79/100.",
    "factors": [
      "Visual patterns strongly suggest manipulation",
      "Unusual frequency distribution (possible synthetic generation)",
      "1 face(s) analyzed for manipulation signs"
    ],
    "recommendation": "Exercise extreme caution. Consider this media potentially misleading."
  },
  "processing_time_ms": 450
}
```

### 5.2 Batch Image Analysis

#### Request
```http
POST /api/v1/analyze/image/batch
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| files | File[] | Yes | Multiple image files (max 10) |

#### Response
```json
{
  "batch_id": "batch-12345",
  "total_files": 5,
  "results": [
    {
      "filename": "image1.jpg",
      "classification": "AUTHENTIC",
      "risk_score": 15.2,
      "confidence": "HIGH"
    },
    {
      "filename": "image2.jpg",
      "classification": "MANIPULATED",
      "risk_score": 82.1,
      "confidence": "HIGH"
    }
  ],
  "processing_time_ms": 2100
}
```

---

## 6. Video Analysis

### Request
```http
POST /api/v1/analyze/video/
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | File | Yes | - | Video file (MP4, AVI, MOV, WebM) |
| num_frames | int | No | 16 | Frames to analyze |

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/video/?num_frames=16" \
  -H "accept: application/json" \
  -F "file=@/path/to/video.mp4"
```

#### Response
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
  "timestamp": "2024-01-15T12:05:00Z",
  "media_type": "video",
  "filename": "test_video.mp4",
  "video_info": {
    "duration_seconds": 30.5,
    "total_frames": 915,
    "fps": 30.0,
    "frames_analyzed": 16
  },
  "classification": "SUSPICIOUS",
  "confidence": "MEDIUM",
  "risk_score": 55.2,
  "average_risk_score": 48.7,
  "frame_analysis": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "risk_score": 42.1,
      "classification": "AUTHENTIC",
      "faces_detected": 1
    },
    {
      "frame_index": 57,
      "timestamp": 1.9,
      "risk_score": 68.5,
      "classification": "SUSPICIOUS",
      "faces_detected": 1
    }
  ],
  "temporal_consistency": "NOT_IMPLEMENTED",
  "processing_time_ms": 8500
}
```

---

## 7. Audio Analysis

### Request
```http
POST /api/v1/analyze/audio/
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Audio file (WAV, MP3, M4A) |

#### Response
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440002",
  "timestamp": "2024-01-15T12:10:00Z",
  "media_type": "audio",
  "filename": "test_audio.wav",
  "audio_info": {
    "duration_seconds": 4.0,
    "sample_rate": 16000
  },
  "classification": "AUTHENTIC",
  "confidence": "MEDIUM",
  "risk_score": 28.5,
  "prediction": {
    "synthetic_probability": 0.285,
    "real_probability": 0.715
  },
  "audio_features": {
    "rms_energy": 0.045,
    "zero_crossing_rate": 0.082
  },
  "processing_time_ms": 1200
}
```

---

## 8. Error Responses

### 400 Bad Request
```json
{
  "detail": "Unsupported file type. Supported: ['.jpg', '.jpeg', '.png', '.webp']"
}
```

### 413 Payload Too Large
```json
{
  "detail": "File too large. Maximum size: 100MB"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Analysis failed: <error message>"
}
```

---

## 9. Response Schema

### Classification Values
| Value | Risk Score Range | Description |
|-------|------------------|-------------|
| `AUTHENTIC` | 0-39 | Likely genuine media |
| `SUSPICIOUS` | 40-69 | Requires caution |
| `MANIPULATED` | 70-100 | Likely fake/altered |

### Confidence Values
| Value | Condition |
|-------|-----------|
| `LOW` | Signal disagreement or near 50% risk |
| `MEDIUM` | Moderate certainty |
| `HIGH` | Strong signal agreement, clear decision |

---

## 10. Rate Limits (Production)

| Endpoint | Limit |
|----------|-------|
| `/analyze/image/` | 100 req/min |
| `/analyze/video/` | 10 req/min |
| `/analyze/audio/` | 50 req/min |
| `/analyze/image/batch` | 10 req/min |

---

## 11. Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

def analyze_image(image_path: str) -> dict:
    """Analyze an image for deepfakes."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/analyze/image/",
            files={"file": f}
        )
    response.raise_for_status()
    return response.json()

def analyze_video(video_path: str, num_frames: int = 16) -> dict:
    """Analyze a video for deepfakes."""
    with open(video_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/analyze/video/",
            files={"file": f},
            params={"num_frames": num_frames}
        )
    response.raise_for_status()
    return response.json()

# Usage
result = analyze_image("test.jpg")
print(f"Classification: {result['classification']}")
print(f"Risk Score: {result['risk_score']}")
```

---

## 12. JavaScript Client Example

```typescript
const API_BASE = "http://localhost:8000/api/v1";

async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/analyze/image/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}
```
