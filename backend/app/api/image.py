"""Image analysis API endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from datetime import datetime
import numpy as np
import cv2
import uuid
import time
import logging

from app.services.visual_detector import visual_detector
from app.services.forensic_analyzer import forensic_analyzer
from app.services.fusion_engine import fusion_engine
from app.services.explainer import explainer
from app.models.response import ImageAnalysisResponse, FaceDetection, Explanation
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in settings.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            400, 
            f"Unsupported format. Supported: {settings.SUPPORTED_IMAGE_FORMATS}"
        )


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from upload file."""
    contents = await file.read()
    
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max: {settings.MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(400, "Could not decode image")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


@router.post("/", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image for deepfake detection.
    
    Uses an Ensemble of:
    1. Visual Xception Detector (Spatial RGB Expert)
    2. Forensic Classifier (Frequency Domain Expert)
    """
    start_time = time.time()
    
    # Validate file
    validate_image_file(file)
    
    # Load image
    image = await load_image_from_upload(file)
    
    logger.info(f"Analyzing image: {file.filename}, shape: {image.shape}")
    
    # 1. Run Visual Analysis (Xception)
    visual_result = visual_detector.analyze(image)
    
    # 2. Run Forensic Analysis
    forensic_result = forensic_analyzer.analyze(image)
    
    # 3. Fuse signals (Ensemble)
    fused_result = fusion_engine.fuse_image_signals(visual_result, forensic_result)
    
    # Generate explanation
    explanation_result = explainer.explain_image_result(
        visual_result, forensic_result, fused_result
    )
    
    # Build face detections
    face_detections = []
    if 'face_predictions' in visual_result:
        for fp in visual_result.get('face_predictions', []):
            face_detections.append(FaceDetection(
                face_id=fp.get('face_id', 0),
                bbox=fp.get('bbox', [0, 0, 0, 0]),
                detection_confidence=fp.get('detection_confidence', 0.0),
                fake_probability=fp.get('fake_probability')
            ))
    
    # Build signals dict
    signals = {}
    
    if visual_result.get('overall_prediction'):
        signals['visual'] = {
            'score': visual_result['overall_prediction'].get('fake_probability', 0.5),
            'weight': 0.6,
            'faces_detected': visual_result.get('faces_detected', 0),
            'analysis_type': visual_result.get('analysis_type', 'xception_visual')
        }
    
    if forensic_result.get('prediction'):
        signals['forensic'] = {
            'score': forensic_result['prediction'].get('fake_probability', 0.5),
            'weight': 0.4,
            'features': forensic_result.get('features', {})
        }
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return ImageAnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        media_type="image",
        filename=file.filename or "unknown",
        classification=fused_result['classification'],
        confidence=fused_result['confidence'],
        risk_score=fused_result['risk_score'],
        signals=signals,
        face_detections=face_detections,
        explanation=Explanation(**explanation_result),
        processing_time_ms=processing_time
    )
