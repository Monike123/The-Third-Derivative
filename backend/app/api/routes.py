"""API routes aggregation."""
from fastapi import APIRouter

from app.api import image, video, audio, advanced, report

router = APIRouter()

# Include all sub-routers
router.include_router(image.router, prefix="/image", tags=["Image Analysis"])
router.include_router(video.router, prefix="/video", tags=["Video Analysis"])
router.include_router(audio.router, prefix="/audio", tags=["Audio Analysis"])
router.include_router(advanced.router, prefix="/advanced", tags=["Enhanced Detection"])
router.include_router(report.router, prefix="/report", tags=["PDF Reports"])
