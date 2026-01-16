"""Audio Detector Service for voice synthesis detection."""
import numpy as np
import onnxruntime as ort
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class AudioDetectorService:
    """Service for audio deepfake/voice synthesis detection."""
    
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self._loaded = False
        
        # Audio params
        self.sample_rate = 16000
        self.duration = 4.0
        self.n_samples = int(self.sample_rate * self.duration)
        self.n_mels = 80
        
    def load_model(self) -> bool:
        """Load audio detector ONNX model."""
        model_path = settings.MODELS_DIR / settings.AUDIO_DETECTOR_PATH
        
        # Try alternative path
        if not model_path.exists():
            model_path = Path("d:/Deepway/Models") / settings.AUDIO_DETECTOR_PATH
        
        if not model_path.exists():
            logger.warning(f"Audio detector model not found at {model_path}")
            return False
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                       if settings.USE_GPU else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self._loaded = True
            logger.info(f"Audio detector loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio detector: {e}")
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)))
            else:
                audio = audio[:self.n_samples]
            
            return audio
            
        except ImportError:
            logger.error("librosa not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        try:
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Add batch and channel dimensions: (1, 1, n_mels, time)
            mel_spec_db = mel_spec_db[np.newaxis, np.newaxis, :, :]
            
            return mel_spec_db.astype(np.float32)
            
        except ImportError:
            logger.error("librosa not installed")
            return np.zeros((1, 1, self.n_mels, 251), dtype=np.float32)
    
    def extract_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract basic audio features for explainability."""
        try:
            import librosa
            
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral centroid
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            
            return {
                'duration': float(len(audio) / self.sample_rate),
                'rms_energy': float(rms),
                'zero_crossing_rate': float(zcr),
                'spectral_centroid': float(spec_cent)
            }
        except:
            return {
                'duration': float(len(audio) / self.sample_rate),
                'rms_energy': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid': 0.0
            }
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file for synthetic voice detection."""
        result = {
            'prediction': None,
            'audio_features': {}
        }
        
        # Load audio
        audio = self.load_audio(audio_path)
        if audio is None:
            result['error'] = 'Failed to load audio'
            result['prediction'] = {'synthetic_probability': 0.5, 'real_probability': 0.5}
            return result
        
        # Extract features for explainability
        result['audio_features'] = self.extract_audio_features(audio)
        
        if not self._loaded:
            result['error'] = 'Model not loaded'
            result['prediction'] = {'synthetic_probability': 0.5, 'real_probability': 0.5}
            return result
        
        try:
            # Extract mel spectrogram
            mel_spec = self.extract_mel_spectrogram(audio)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: mel_spec})
            logits = outputs[0][0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            synthetic_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            
            result['prediction'] = {
                'synthetic_probability': synthetic_prob,
                'real_probability': 1.0 - synthetic_prob
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            result['error'] = str(e)
            result['prediction'] = {'synthetic_probability': 0.5, 'real_probability': 0.5}
        
        return result
    
    def analyze_from_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Analyze audio from bytes (for file upload)."""
        import tempfile
        import os
        
        # Get file extension
        ext = os.path.splitext(filename)[1] or '.wav'
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            result = self.analyze(tmp_path)
        finally:
            os.unlink(tmp_path)
        
        return result


# Singleton instance
audio_detector = AudioDetectorService()
