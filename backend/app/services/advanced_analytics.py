"""Enhanced Detection Service - Advanced Deep Learning Models (Dual-Model Ensemble)."""
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedAnalyticsService:
    """
    Enhanced deepfake detection using DUAL cloud-based models for maximum accuracy.
    Runs two different models and averages the results.
    """
    
    # Internal model references
    _IMAGE_MODEL_V2 = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    _IMAGE_MODEL_V3 = "prithivMLmods/deepfake-detector-model-v1"  # SigLIP-based
    
    # Public names (shown to users)
    IMAGE_MODEL = "DeepVision-Ensemble (v2+v3)"
    AUDIO_MODEL = "AudioForensics-v2"
    
    def __init__(self):
        self.client = None
        self._available = False
        self.api_key = os.environ.get("HF_TOKEN")
        
    def load(self) -> bool:
        """Initialize the HuggingFace Inference Client."""
        if not self.api_key:
            logger.warning("HF_TOKEN not set - Advanced Analytics unavailable")
            return False
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=self.api_key,
            )
            self._available = True
            logger.info("Advanced Analytics (Dual-Model Ensemble) initialized")
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if advanced analytics is available."""
        return self._available
    
    def _parse_v2_output(self, output) -> tuple:
        """Parse V2 model output (Fake/Real labels)."""
        results = {item['label']: item['score'] for item in output}
        fake_prob = results.get('Fake', results.get('fake', results.get('AI', 0)))
        real_prob = results.get('Real', results.get('real', results.get('Human', 0)))
        
        if fake_prob == 0 and real_prob == 0 and output:
            first_label = output[0]['label'].lower()
            first_score = output[0]['score']
            if 'fake' in first_label or 'ai' in first_label:
                fake_prob, real_prob = first_score, 1 - first_score
            else:
                real_prob, fake_prob = first_score, 1 - first_score
        
        return fake_prob, real_prob
    
    def _parse_v3_output(self, output) -> tuple:
        """Parse V3 model output (Class 0=fake, Class 1=real - INVERTED!)."""
        results = {item['label']: item['score'] for item in output}
        
        # V3 uses: fake (class 0), real (class 1)
        fake_prob = results.get('fake', results.get('Fake', 0))
        real_prob = results.get('real', results.get('Real', 0))
        
        # Handle numeric labels if model returns 0/1
        if fake_prob == 0 and real_prob == 0:
            fake_prob = results.get('0', results.get('LABEL_0', 0))
            real_prob = results.get('1', results.get('LABEL_1', 0))
        
        return fake_prob, real_prob
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using DUAL model ensemble for maximum accuracy."""
        if not self._available:
            return {'error': 'Enhanced detection unavailable', 'available': False}
        
        v2_result = None
        v3_result = None
        v2_fake, v2_real = 0.5, 0.5
        v3_fake, v3_real = 0.5, 0.5
        
        # Run Model V2
        try:
            output_v2 = self.client.image_classification(image_path, model=self._IMAGE_MODEL_V2)
            v2_fake, v2_real = self._parse_v2_output(output_v2)
            v2_result = output_v2
            logger.info(f"V2 Model: fake={v2_fake:.3f}, real={v2_real:.3f}")
        except Exception as e:
            logger.warning(f"V2 model failed: {e}")
        
        # Run Model V3 (SigLIP-based)
        try:
            output_v3 = self.client.image_classification(image_path, model=self._IMAGE_MODEL_V3)
            v3_fake, v3_real = self._parse_v3_output(output_v3)
            v3_result = output_v3
            logger.info(f"V3 Model: fake={v3_fake:.3f}, real={v3_real:.3f}")
        except Exception as e:
            logger.warning(f"V3 model failed: {e}")
        
        # Ensemble: Average both models (weighted can be tuned)
        if v2_result and v3_result:
            # Both models available - average
            fake_prob = (v2_fake * 0.5) + (v3_fake * 0.5)
            real_prob = (v2_real * 0.5) + (v3_real * 0.5)
            models_used = "V2+V3 Ensemble"
        elif v2_result:
            fake_prob, real_prob = v2_fake, v2_real
            models_used = "V2 Only"
        elif v3_result:
            fake_prob, real_prob = v3_fake, v3_real
            models_used = "V3 Only"
        else:
            return {'error': 'Both models failed', 'available': False}
        
        risk_score = fake_prob * 100
        
        # Confidence and Classification
        if real_prob >= 0.90:
            classification = 'AUTHENTIC'
            confidence = 'HIGH'
        elif fake_prob >= 0.80:
            classification = 'MANIPULATED'
            confidence = 'HIGH'
        elif risk_score >= 70:
            classification = 'MANIPULATED'
            confidence = 'MEDIUM'
        elif risk_score >= 40:
            classification = 'SUSPICIOUS'
            confidence = 'MEDIUM'
        else:
            classification = 'AUTHENTIC'
            confidence = 'MEDIUM' if abs(risk_score - 50) > 20 else 'LOW'
        
        return {
            'model': self.IMAGE_MODEL,
            'models_used': models_used,
            'raw_output': {
                'v2': v2_result,
                'v3': v3_result
            },
            'prediction': {
                'fake_probability': round(fake_prob, 4),
                'real_probability': round(real_prob, 4),
                'v2_fake': round(v2_fake, 4),
                'v3_fake': round(v3_fake, 4)
            },
            'risk_score': round(risk_score, 2),
            'classification': classification,
            'confidence': confidence
        }
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Audio analysis uses local models only."""
        return {
            'error': 'Enhanced audio detection uses local processing. Use Standard Analysis.',
            'available': False,
            'use_basic': True
        }


# Singleton
advanced_analytics = AdvancedAnalyticsService()

