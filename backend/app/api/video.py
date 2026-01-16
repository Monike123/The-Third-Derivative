"""Video analysis API endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from datetime import datetime
import numpy as np
import cv2
import uuid
import time
import tempfile
import os
import logging

from app.services.temporal_detector import temporal_detector
from app.services.visual_detector import visual_detector
from app.services.forensic_analyzer import forensic_analyzer
from app.services.audio_detector import audio_detector
from app.services.fusion_engine import fusion_engine
from app.services.explainer import explainer
from app.models.response import VideoAnalysisResponse, FrameAnalysis, Explanation
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            400, 
            f"Unsupported format. Supported: {settings.SUPPORTED_VIDEO_FORMATS}"
        )


def extract_frames(video_path: str, num_frames: int = 16) -> tuple:
    """Extract frames from video uniformly."""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_info = {
        'duration_seconds': round(duration, 2),
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'resolution': f"{width}x{height}",
        'frames_analyzed': num_frames
    }
    
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames = []
    frame_timestamps = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_timestamps.append(idx / fps)
    
    cap.release()
    
    return frames, frame_timestamps, video_info


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio track from video using ffmpeg."""
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        logger.warning(f"Failed to extract audio: {e}")
        return False


@router.post("/", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    num_frames: int = Query(default=16, ge=4, le=32, description="Number of frames to analyze")
):
    """
    Analyze a video for deepfake detection.
    
    Uses Temporal Convolutional Networks (3D CNN) to detect motion and temporal inconsistencies.
    Also applies Forensic analysis on key frames.
    """
    start_time = time.time()
    
    # Validate file
    validate_video_file(file)
    
    # Save to temp file
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else '.mp4'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(413, f"File too large. Max: {settings.MAX_FILE_SIZE // (1024*1024)}MB")
        tmp.write(contents)
        video_path = tmp.name
    
    audio_path = video_path.replace(ext, '.wav')
    
    try:
        logger.info(f"Analyzing video: {file.filename}")
        
        # 1. Run Temporal Analysis (Main Detector)
        temporal_result = temporal_detector.analyze(video_path)
        
        # Extract frames to populate metadata and basic frame analysis
        frames, timestamps, video_info = extract_frames(video_path, num_frames)
        
        # 2. Analyze individual frames (Spatial + Frequency Ensemble)
        frame_analyses = []
        frame_results = []
        
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            # Visual analysis (Spatial)
            visual_result = visual_detector.analyze(frame)
            
            # Forensic analysis (Frequency)
            forensic_result = forensic_analyzer.analyze(frame)
            
            # Fuse for this frame (Ensemble)
            fused = fusion_engine.fuse_image_signals(visual_result, forensic_result)
            
            frame_analyses.append(FrameAnalysis(
                frame_index=i,
                timestamp_seconds=round(ts, 2),
                risk_score=fused['risk_score'],
                classification=fused['classification'],
                faces_detected=visual_result.get('faces_detected', 0)
            ))
            
            frame_results.append({
                'risk_score': fused['risk_score'],
                'classification': fused['classification']
            })
        
        # 3. Audio Analysis
        audio_result = None
        if extract_audio(video_path, audio_path):
            audio_result = audio_detector.analyze(audio_path)
        
        # 4. Final Aggregation
        # We prioritize Temporal Detector result as it's the requested standard
        
        if 'error' in temporal_result:
             # Fallback to forensic average if temporal fails
             logger.warning("Temporal analysis failed, falling back to forensic average")
             avg_risk = np.mean([f.risk_score for f in frame_analyses]) if frame_analyses else 0
             classification = 'SUSPICIOUS' if avg_risk > 50 else 'AUTHENTIC'
             risk_score = avg_risk
             confidence = 'LOW'
        else:
             classification = temporal_result['classification']
             risk_score = temporal_result['risk_score']
             confidence = temporal_result['confidence']
        
        # Generate explanation
        explanation_result = explainer.explain_video_result(
            len(frames), 
            {'classification': classification, 'risk_score': risk_score}, 
            audio_result
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return VideoAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            media_type="video",
            filename=file.filename or "unknown",
            video_info=video_info,
            classification=classification,
            confidence=confidence,
            risk_score=risk_score,
            average_risk_score=round(risk_score, 2),
            frame_analysis=frame_analyses,
            temporal_consistency="HIGH" if classification == 'AUTHENTIC' else "LOW",
            audio_analysis=audio_result,
            explanation=Explanation(**explanation_result),
            processing_time_ms=processing_time
        )
        
    finally:
        # Cleanup temp files
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)
