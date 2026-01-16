"""Advanced Analytics API endpoint using HuggingFace models."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from datetime import datetime
from typing import Optional
import uuid
import time
import tempfile
import os
import logging

from app.services.advanced_analytics import advanced_analytics
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def get_advanced_status():
    """Check if Enhanced Detection is available."""
    from app.services.vera_ai import vera_ai
    return {
        "available": advanced_analytics.is_available(),
        "image_model": advanced_analytics.IMAGE_MODEL,
        "audio_model": advanced_analytics.AUDIO_MODEL,
        "forensic_available": vera_ai.is_available()
    }


@router.post("/image/")
async def analyze_image_advanced(file: UploadFile = File(...)):
    """
    Analyze image using DeepVision enhanced detection model.
    
    Uses our advanced deep learning model for enhanced accuracy.
    
    - **file**: Image file (JPEG, PNG, WebP)
    """
    start_time = time.time()
    
    if not advanced_analytics.is_available():
        raise HTTPException(
            503,
            "Advanced Analytics not available. Set HF_TOKEN environment variable."
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in settings.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    # Read and save temp file
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Advanced Analytics analyzing: {file.filename}")
        
        result = advanced_analytics.analyze_image(tmp_path)
        
        if 'error' in result and not result.get('prediction'):
            raise HTTPException(500, f"Analysis failed: {result['error']}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "media_type": "image",
            "filename": file.filename,
            "mode": "advanced",
            "model": advanced_analytics.IMAGE_MODEL,
            "classification": result.get('classification', 'UNKNOWN'),
            "confidence": result.get('confidence', 'MEDIUM'),
            "risk_score": result.get('risk_score', 50),
            "prediction": result.get('prediction', {}),
            "raw_output": result.get('raw_output', []),
            "processing_time_ms": processing_time
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/compare/")
async def compare_analysis(file: UploadFile = File(...)):
    """
    Run both Basic and Advanced analytics and compare results.
    
    Returns side-by-side comparison of local models vs HuggingFace model.
    """
    start_time = time.time()
    
    # Validate
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in settings.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    contents = await file.read()
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        import numpy as np
        import cv2
        from app.services.visual_detector import visual_detector
        from app.services.forensic_analyzer import forensic_analyzer
        from app.services.fusion_engine import fusion_engine
        
        # Load image for basic analysis
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic analysis
        visual_result = visual_detector.analyze(image)
        forensic_result = forensic_analyzer.analyze(image)
        basic_fused = fusion_engine.fuse_image_signals(visual_result, forensic_result)
        
        # Advanced analysis (if available)
        advanced_result = None
        if advanced_analytics.is_available():
            advanced_result = advanced_analytics.analyze_image(tmp_path)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "comparison": {
                "basic": {
                    "classification": basic_fused['classification'],
                    "risk_score": basic_fused['risk_score'],
                    "confidence": basic_fused['confidence'],
                    "signals": basic_fused.get('signal_values', {})
                },
                "advanced": {
                    "available": advanced_analytics.is_available(),
                    "classification": advanced_result.get('classification') if advanced_result else None,
                    "risk_score": advanced_result.get('risk_score') if advanced_result else None,
                    "confidence": advanced_result.get('confidence') if advanced_result else None,
                    "model": advanced_analytics.IMAGE_MODEL
                }
            },
            "processing_time_ms": processing_time
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/audio/")
async def analyze_audio_advanced(file: UploadFile = File(...)):
    """
    Analyze audio using HuggingFace Deepfake-audio-detection-V2.
    
    Cloud-based inference for synthetic voice detection.
    Requires HF_TOKEN environment variable.
    
    - **file**: Audio file (WAV, MP3, M4A, FLAC)
    """
    start_time = time.time()
    
    if not advanced_analytics.is_available():
        raise HTTPException(
            503,
            "Advanced Analytics not available. Set HF_TOKEN environment variable."
        )
    
    # Validate
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Advanced Audio Analytics: {file.filename}")
        
        result = advanced_analytics.analyze_audio(tmp_path)
        
        if 'error' in result and not result.get('prediction'):
            raise HTTPException(500, f"Analysis failed: {result['error']}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "media_type": "audio",
            "filename": file.filename,
            "mode": "advanced",
            "model": advanced_analytics.AUDIO_MODEL,
            "classification": result.get('classification', 'UNKNOWN'),
            "confidence": result.get('confidence', 'MEDIUM'),
            "risk_score": result.get('risk_score', 50),
            "prediction": result.get('prediction', {}),
            "raw_output": result.get('raw_output', []),
            "processing_time_ms": processing_time
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/video/")
async def analyze_video_advanced(
    file: UploadFile = File(...),
    max_frames: int = Query(5, ge=1, le=10, description="Max frames to analyze")
):
    """
    Analyze video using DeepVision enhanced detection model.
    
    Extracts key frames and analyzes each using our cloud-based model.
    
    - **file**: Video file (MP4, AVI, MOV, WebM)
    - **max_frames**: Number of frames to analyze (1-10)
    """
    import cv2
    import numpy as np
    
    start_time = time.time()
    
    if not advanced_analytics.is_available():
        raise HTTPException(
            503,
            "Advanced Analytics not available. Set HF_TOKEN environment variable."
        )
    
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
        raise HTTPException(400, f"Unsupported video format: {ext}")
    
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        video_path = tmp.name
    
    frame_results = []
    temp_frames = []
    
    try:
        logger.info(f"Advanced Video Analytics: {file.filename}")
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise HTTPException(400, "Could not read video frames")
        
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Save frame temporarily
                frame_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                cv2.imwrite(frame_path, frame)
                temp_frames.append(frame_path)
                
                # Analyze frame with cloud model
                result = advanced_analytics.analyze_image(frame_path)
                
                if 'error' not in result:
                    frame_results.append({
                        'frame_index': int(idx),
                        'classification': result.get('classification', 'UNKNOWN'),
                        'risk_score': result.get('risk_score', 50),
                        'confidence': result.get('confidence', 'MEDIUM')
                    })
        
        cap.release()
        
        if not frame_results:
            raise HTTPException(500, "Failed to analyze video frames")
        
        # Aggregate results
        avg_risk = sum(f['risk_score'] for f in frame_results) / len(frame_results)
        manipulated_count = sum(1 for f in frame_results if f['classification'] == 'MANIPULATED')
        suspicious_count = sum(1 for f in frame_results if f['classification'] == 'SUSPICIOUS')
        
        # Determine overall classification (Stricter thresholds for MANIPULATED)
        if avg_risk >= 75 or (manipulated_count >= len(frame_results) * 0.7):
            classification = 'MANIPULATED'
        elif avg_risk >= 55 or (manipulated_count + suspicious_count >= len(frame_results) * 0.5):
            classification = 'SUSPICIOUS'
        else:
            classification = 'AUTHENTIC'
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "media_type": "video",
            "filename": file.filename,
            "mode": "advanced",
            "model": advanced_analytics.IMAGE_MODEL,
            "classification": classification,
            "confidence": "HIGH" if abs(avg_risk - 50) > 30 else "MEDIUM",
            "risk_score": round(avg_risk, 2),
            "prediction": {
                "fake_probability": avg_risk / 100,
                "real_probability": 1 - (avg_risk / 100)
            },
            "frame_analysis": frame_results,
            "frames_analyzed": len(frame_results),
            "total_frames": total_frames,
            "processing_time_ms": processing_time
        }
        
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        for fp in temp_frames:
            if os.path.exists(fp):
                os.unlink(fp)
