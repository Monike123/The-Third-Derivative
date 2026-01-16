"""Configuration settings for the Deepfake Detection API."""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Model Paths - Updated to use the Models folder
    MODELS_DIR: Path = Path(__file__).parent.parent.parent / "Models"
    VISUAL_DETECTOR_PATH: str = "visual_detector.onnx"
    FORENSIC_CLASSIFIER_PATH: str = "forensic_classifier.onnx"
    FORENSIC_SCALER_PATH: str = "forensic_scaler.pkl"
    AUDIO_DETECTOR_PATH: str = "audio_detector.onnx"
    TEMPORAL_DETECTOR_PATH: str = "temporal_deepfake_detector.onnx"
    
    # Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_IMAGE_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    SUPPORTED_VIDEO_FORMATS: List[str] = [".mp4", ".avi", ".mov", ".webm", ".mkv"]
    SUPPORTED_AUDIO_FORMATS: List[str] = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    
    # Inference
    USE_GPU: bool = False  # Set to True if you have GPU
    
    # Fusion weights
    VISUAL_WEIGHT: float = 0.45
    FORENSIC_WEIGHT: float = 0.30
    TEMPORAL_WEIGHT: float = 0.15
    AUDIO_WEIGHT: float = 0.10
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
