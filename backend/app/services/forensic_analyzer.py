"""Forensic Analyzer Service using extracted image features."""
import numpy as np
import cv2
import onnxruntime as ort
from scipy import fftpack
from scipy.stats import entropy
import joblib
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class ForensicAnalyzerService:
    """Service for forensic feature analysis of images."""
    
    # Feature names for explainability
    FEATURE_NAMES = [
        'high_freq_ratio', 'spectral_entropy', 'magnitude_mean', 'magnitude_std',
        'noise_variance', 'noise_mean', 'noise_var_r', 'noise_var_g',
        'noise_var_b', 'noise_var_ratio', 'sharpness', 'edge_density'
    ]
    
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.scaler = None
        self._loaded = False
        
    def load_model(self) -> bool:
        """Load forensic classifier and scaler."""
        model_path = settings.MODELS_DIR / settings.FORENSIC_CLASSIFIER_PATH
        scaler_path = settings.MODELS_DIR / settings.FORENSIC_SCALER_PATH
        
        # Try alternative paths
        if not model_path.exists():
            model_path = Path("d:/Deepway/Models") / settings.FORENSIC_CLASSIFIER_PATH
        if not scaler_path.exists():
            scaler_path = Path("d:/Deepway/Models") / settings.FORENSIC_SCALER_PATH
        
        try:
            if model_path.exists():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                           if settings.USE_GPU else ['CPUExecutionProvider']
                self.session = ort.InferenceSession(str(model_path), providers=providers)
                logger.info(f"Forensic classifier loaded from {model_path}")
            else:
                logger.warning(f"Forensic classifier not found at {model_path}")
                
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Forensic scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Forensic scaler not found at {scaler_path}")
            
            self._loaded = self.session is not None and self.scaler is not None
            return self._loaded
            
        except Exception as e:
            logger.error(f"Failed to load forensic analyzer: {e}")
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def extract_fft_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        
        # 2D FFT
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High frequency ratio (energy in outer regions vs total)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        low_freq = magnitude[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        high_freq_ratio = 1 - (np.sum(low_freq) / (np.sum(magnitude) + 1e-10))
        
        # Spectral entropy
        magnitude_flat = magnitude.flatten()
        magnitude_norm = magnitude_flat / (np.sum(magnitude_flat) + 1e-10)
        spectral_entropy_val = entropy(magnitude_norm + 1e-10)
        
        return {
            'high_freq_ratio': float(high_freq_ratio),
            'spectral_entropy': float(spectral_entropy_val),
            'magnitude_mean': float(np.mean(magnitude)),
            'magnitude_std': float(np.std(magnitude))
        }
    
    def extract_noise_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract noise residual features."""
        # Use Gaussian blur instead of NLMeans for speed
        image_resized = cv2.resize(image, (256, 256))
        blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
        noise = image_resized.astype(np.float32) - blurred.astype(np.float32)
        
        # Overall noise statistics
        noise_var = float(np.var(noise))
        noise_mean = float(np.mean(np.abs(noise)))
        
        # Per-channel variance
        noise_var_r = float(np.var(noise[:, :, 0]))
        noise_var_g = float(np.var(noise[:, :, 1]))
        noise_var_b = float(np.var(noise[:, :, 2]))
        
        # Ratio of max to min channel variance
        min_var = min(noise_var_r, noise_var_g, noise_var_b)
        max_var = max(noise_var_r, noise_var_g, noise_var_b)
        noise_var_ratio = max_var / (min_var + 1e-10)
        
        return {
            'noise_variance': noise_var,
            'noise_mean': noise_mean,
            'noise_var_r': noise_var_r,
            'noise_var_g': noise_var_g,
            'noise_var_b': noise_var_b,
            'noise_var_ratio': float(noise_var_ratio)
        }
    
    def extract_quality_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract image quality features."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        return {
            'sharpness': float(laplacian_var),
            'edge_density': float(edge_density)
        }
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract all forensic features from an image."""
        features = {}
        features.update(self.extract_fft_features(image))
        features.update(self.extract_noise_features(image))
        features.update(self.extract_quality_features(image))
        return features
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """Full forensic analysis of an image."""
        # Extract features
        features = self.extract_all_features(image)
        
        result = {
            'features': features,
            'prediction': None,
            'feature_importance': {}
        }
        
        if not self._loaded:
            # Return features only, with default prediction
            result['error'] = 'Model not loaded - returning features only'
            result['prediction'] = {'fake_probability': 0.5, 'real_probability': 0.5}
            return result
        
        try:
            # Prepare feature vector in correct order
            feature_vector = np.array([[
                features.get(name, 0.0) for name in self.FEATURE_NAMES
            ]], dtype=np.float32)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run(
                [output_name], 
                {input_name: feature_vector_scaled.astype(np.float32)}
            )
            logits = outputs[0][0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            
            result['prediction'] = {
                'fake_probability': fake_prob,
                'real_probability': 1.0 - fake_prob
            }
            
            # Feature importance (deviation from mean in scaled space)
            for i, name in enumerate(self.FEATURE_NAMES):
                deviation = abs(feature_vector_scaled[0][i])
                result['feature_importance'][name] = float(deviation)
            
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            result['error'] = str(e)
            result['prediction'] = {'fake_probability': 0.5, 'real_probability': 0.5}
        
        return result


# Singleton instance
forensic_analyzer = ForensicAnalyzerService()
