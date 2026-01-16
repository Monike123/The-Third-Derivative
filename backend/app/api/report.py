"""PDF Report API endpoint."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import io
import logging

from app.services.pdf_generator import pdf_generator

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalysisResultInput(BaseModel):
    """Input model for PDF report generation."""
    analysis_id: str
    timestamp: str
    media_type: str = "image"
    filename: str
    mode: Optional[str] = "advanced"
    model: Optional[str] = "DeepVision-v2"
    classification: str
    confidence: str
    risk_score: float
    prediction: Dict[str, float]
    raw_output: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[int] = 0


@router.post("/image")
async def generate_image_report(data: AnalysisResultInput):
    """
    Generate a detailed PDF report for image analysis results.
    
    - **data**: Analysis result from the detection API
    
    Returns: PDF file download
    """
    try:
        # Convert to dict
        analysis_data = data.model_dump()
        
        # Generate PDF
        pdf_bytes = pdf_generator.generate_image_report(analysis_data)
        
        # Create filename
        safe_filename = data.filename.replace(' ', '_').replace('.', '_')
        report_filename = f"deepfake_report_{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={report_filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(500, f"Failed to generate PDF report: {str(e)}")


@router.post("/from-json")
async def generate_report_from_json(analysis_result: Dict[str, Any]):
    """
    Generate PDF report from raw JSON analysis result.
    
    Accepts the raw JSON output from any analysis endpoint.
    """
    try:
        pdf_bytes = pdf_generator.generate_image_report(analysis_result)
        
        filename = analysis_result.get('filename', 'analysis')
        safe_filename = filename.replace(' ', '_').replace('.', '_')
        report_filename = f"deepfake_report_{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={report_filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(500, f"Failed to generate PDF report: {str(e)}")
