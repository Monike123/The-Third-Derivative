"""Vera-AI Forensic Detection Service using HuggingFace Space."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class VeraAIService:
    """
    Forensic deepfake detection using Vera-AI-Secure HuggingFace Space.
    Uses Gradio Client to call the Space API.
    
    Space: https://huggingface.co/spaces/Dileep07/Vera-AI-Secure
    """
    
    SPACE_ID = "Dileep07/Vera-AI-Secure"
    
    def __init__(self):
        self.client = None
        self._available = False
        
    def load(self) -> bool:
        """Initialize the Gradio Client for Vera-AI Space."""
        try:
            from gradio_client import Client
            self.client = Client(self.SPACE_ID)
            self._available = True
            logger.info(f"Vera-AI Forensic Service initialized ({self.SPACE_ID})")
            return True
        except ImportError:
            logger.warning("gradio_client not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Vera-AI Space: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Vera-AI service is available."""
        return self._available
    
    def analyze_image(self, image_path: str, name: str = "User", case_ref: str = "API-001") -> Dict[str, Any]:
        """
        Analyze image using Vera-AI forensic detection.
        
        Args:
            image_path: Path to the image file
            name: User name for the report
            case_ref: Case reference number
            
        Returns:
            Dict with forensic analysis results
        """
        if not self._available:
            return {
                'error': 'Vera-AI service not available',
                'available': False
            }
        
        try:
            # Call the Vera-AI Space
            # The Space expects: image, name, case_reference
            result = self.client.predict(
                image_path,  # Input image
                name,        # Name
                case_ref,    # Case Reference
                api_name="/predict"  # Adjust based on actual API
            )
            
            # Parse the result
            # Vera-AI returns a forensic report
            if isinstance(result, str):
                # Text report
                is_fake = any(word in result.lower() for word in ['fake', 'manipulated', 'deepfake', 'tampered'])
                risk_score = 75 if is_fake else 25
                classification = 'MANIPULATED' if is_fake else 'AUTHENTIC'
            elif isinstance(result, dict):
                # Structured result
                is_fake = result.get('is_fake', result.get('prediction', 'real')).lower() in ['fake', 'true', '1']
                risk_score = result.get('confidence', 0.75 if is_fake else 0.25) * 100
                classification = 'MANIPULATED' if is_fake else 'AUTHENTIC'
            else:
                # Tuple or list result (common for Gradio)
                report = str(result)
                is_fake = any(word in report.lower() for word in ['fake', 'manipulated', 'deepfake', 'tampered'])
                risk_score = 75 if is_fake else 25
                classification = 'MANIPULATED' if is_fake else 'AUTHENTIC'
            
            return {
                'model': 'Vera-AI-Secure',
                'space_id': self.SPACE_ID,
                'raw_output': str(result)[:2000],  # Truncate long reports
                'classification': classification,
                'risk_score': round(risk_score, 2),
                'confidence': 'HIGH' if abs(risk_score - 50) > 30 else 'MEDIUM',
                'forensic_report': str(result) if isinstance(result, str) else None
            }
            
        except Exception as e:
            logger.error(f"Vera-AI analysis failed: {e}")
            return {
                'error': str(e),
                'space_id': self.SPACE_ID
            }


# Singleton
vera_ai = VeraAIService()
