"""Response models for the Deepfake Detection API."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class FaceDetection(BaseModel):
    """Face detection result."""
    face_id: int
    bbox: List[int] = Field(description="[x1, y1, x2, y2]")
    detection_confidence: float
    fake_probability: Optional[float] = None


class SignalResult(BaseModel):
    """Individual signal result."""
    score: float
    weight: float
    details: Optional[Dict[str, Any]] = None


class Explanation(BaseModel):
    """Human-readable explanation."""
    summary: str
    factors: List[str]
    recommendation: str


class ImageAnalysisResponse(BaseModel):
    """Response for image analysis."""
    analysis_id: str
    timestamp: datetime
    media_type: str = "image"
    filename: str
    
    # Core results
    classification: str = Field(description="AUTHENTIC, SUSPICIOUS, or MANIPULATED")
    confidence: str = Field(description="LOW, MEDIUM, or HIGH")
    risk_score: float = Field(ge=0, le=100)
    
    # Detailed signals
    signals: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Face detections
    face_detections: List[FaceDetection] = Field(default_factory=list)
    
    # Explanation
    explanation: Explanation
    
    # Performance
    processing_time_ms: int


class FrameAnalysis(BaseModel):
    """Per-frame analysis result."""
    frame_index: int
    timestamp_seconds: float
    risk_score: float
    classification: str
    faces_detected: int


class VideoAnalysisResponse(BaseModel):
    """Response for video analysis."""
    analysis_id: str
    timestamp: datetime
    media_type: str = "video"
    filename: str
    
    # Video info
    video_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Core results
    classification: str
    confidence: str
    risk_score: float = Field(ge=0, le=100)
    average_risk_score: float
    
    # Frame-level analysis
    frame_analysis: List[FrameAnalysis] = Field(default_factory=list)
    
    # Additional signals
    temporal_consistency: Optional[str] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    
    # Explanation
    explanation: Explanation
    
    processing_time_ms: int


class AudioAnalysisResponse(BaseModel):
    """Response for audio analysis."""
    analysis_id: str
    timestamp: datetime
    media_type: str = "audio"
    filename: str
    
    # Audio info
    audio_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Core results
    classification: str
    confidence: str
    risk_score: float = Field(ge=0, le=100)
    
    # Prediction details
    prediction: Dict[str, float]
    audio_features: Dict[str, float]
    
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None
