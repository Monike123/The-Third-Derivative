"""
Deepfake Detection API - Main Application
==========================================

FastAPI application for detecting deepfake images, videos, and synthetic audio.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.api import routes
from app.config import settings
from app.services.visual_detector import visual_detector
from app.services.forensic_analyzer import forensic_analyzer
from app.services.audio_detector import audio_detector
from app.services.fusion_engine import fusion_engine
from app.services.advanced_analytics import advanced_analytics
from app.models.response import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load models on startup."""
    logger.info("=" * 50)
    logger.info("Deepfake Detection API Starting...")
    logger.info("=" * 50)
    
    # Load models
    logger.info("Loading ML models...")
    
    visual_loaded = visual_detector.load_model()
    logger.info(f"  Visual Detector: {'✓' if visual_loaded else '✗ (not found)'}")
    
    forensic_loaded = forensic_analyzer.load_model()
    logger.info(f"  Forensic Analyzer: {'✓' if forensic_loaded else '✗ (not found)'}")
    
    audio_loaded = audio_detector.load_model()
    logger.info(f"  Audio Detector: {'✓' if audio_loaded else '✗ (not found)'}")
    
    fusion_engine.initialize()
    logger.info(f"  Fusion Engine: ✓")
    
    # Advanced Analytics (HuggingFace)
    advanced_loaded = advanced_analytics.load()
    logger.info(f"  Advanced Analytics: {'✓' if advanced_loaded else '✗ (HF_TOKEN not set)'}")
    
    logger.info("=" * 50)
    logger.info(f"Models directory: {settings.MODELS_DIR}")
    logger.info(f"GPU enabled: {settings.USE_GPU}")
    logger.info("=" * 50)
    logger.info("API Ready!")
    logger.info(f"Docs: http://localhost:{settings.API_PORT}/docs")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Shutting down Deepfake Detection API...")


# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="""
    # Deepfake Detection & Media Authenticity Analyzer
    
    Advanced AI system for detecting manipulated media including:
    - **Image Analysis**: Detect deepfake images using visual and forensic signals
    - **Video Analysis**: Analyze videos frame-by-frame with optional audio analysis
    - **Audio Analysis**: Detect synthetic/AI-generated voice
    
    ## Risk Scores
    - **0-39**: AUTHENTIC - Likely genuine content
    - **40-69**: SUSPICIOUS - Requires caution
    - **70-100**: MANIPULATED - Likely fake/altered
    
    ## Confidence Levels
    - **HIGH**: Strong signal agreement, clear detection
    - **MEDIUM**: Moderate certainty
    - **LOW**: Signals disagree or near threshold
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Deepfake Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "image": "/api/v1/analyze/image/",
            "video": "/api/v1/analyze/video/",
            "audio": "/api/v1/analyze/audio/",
            "advanced": "/api/v1/analyze/advanced/image/"
        },
        "advanced_analytics": advanced_analytics.is_available()
    }


# Health check
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "visual_detector": visual_detector.is_loaded(),
            "forensic_analyzer": forensic_analyzer.is_loaded(),
            "audio_detector": audio_detector.is_loaded(),
            "advanced_analytics": advanced_analytics.is_available()
        },
        version="1.0.0"
    )


# Include API routes
app.include_router(routes.router, prefix="/api/v1/analyze")


# Run with: uvicorn app.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
