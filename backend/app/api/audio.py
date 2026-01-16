"""Audio analysis API endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from datetime import datetime
import uuid
import time
import tempfile
import os
import logging

from app.services.audio_detector import audio_detector
from app.models.response import AudioAnalysisResponse
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in settings.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            400, 
            f"Unsupported format. Supported: {settings.SUPPORTED_AUDIO_FORMATS}"
        )


@router.post("/", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for synthetic voice detection.
    
    - **file**: Audio file (WAV, MP3, M4A, FLAC, OGG)
    
    Returns:
    - Risk score (0-100)
    - Classification
    - Audio features
    """
    start_time = time.time()
    
    # Validate file
    validate_audio_file(file)
    
    # Save to temp file
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else '.wav'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(413, f"File too large. Max: {settings.MAX_FILE_SIZE // (1024*1024)}MB")
        tmp.write(contents)
        audio_path = tmp.name
    
    try:
        logger.info(f"Analyzing audio: {file.filename}")
        
        # Run audio analysis
        result = audio_detector.analyze(audio_path)
        
        if 'error' in result and not result.get('prediction'):
            raise HTTPException(500, f"Audio analysis failed: {result['error']}")
        
        prediction = result.get('prediction', {})
        audio_features = result.get('audio_features', {})
        
        # Compute risk score
        synthetic_prob = prediction.get('synthetic_probability', 0.5)
        risk_score = synthetic_prob * 100
        
        # Classification
        if risk_score >= 70:
            classification = 'SYNTHETIC'
        elif risk_score >= 40:
            classification = 'SUSPICIOUS'
        else:
            classification = 'AUTHENTIC'
        
        # Confidence
        distance_from_uncertain = abs(risk_score - 50)
        if distance_from_uncertain > 35:
            confidence = 'HIGH'
        elif distance_from_uncertain > 15:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return AudioAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            media_type="audio",
            filename=file.filename or "unknown",
            audio_info={
                'duration_seconds': audio_features.get('duration', 0),
                'sample_rate': 16000
            },
            classification=classification,
            confidence=confidence,
            risk_score=round(risk_score, 2),
            prediction=prediction,
            audio_features=audio_features,
            processing_time_ms=processing_time
        )
        
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.unlink(audio_path)
