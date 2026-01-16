"""Ensemble detector using Xception + EfficientNet-B7 for high accuracy."""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import xception, Xception_Weights, efficientnet_b7, EfficientNet_B7_Weights
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble detector combining Xception and EfficientNet-B7.
    Uses weighted voting for final prediction.
    """
    
    def __init__(self):
        self.xception_model = None
        self.efficientnet_model = None
        self.device = torch.device('cuda' if settings.USE_GPU and torch.cuda.is_available() else 'cpu')
        self._loaded = False
        self.input_size = (299, 299)  # Xception input size
        
        # Image preprocessing
        self.xception_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.efficientnet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((600, 600)),  # EfficientNet-B7 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ensemble weights (can be tuned)
        self.xception_weight = 0.5
        self.efficientnet_weight = 0.5
        
    def load_model(self) -> bool:
        """Load pretrained ensemble models (Xception and EfficientNet-B7)."""
        try:
            # Load pretrained model (using same pretrained model for both)
            ensemble_weights_path = settings.MODELS_DIR / settings.PRETRAINED_ENSEMBLE_MODEL_PATH
            xception_weights_path = ensemble_weights_path  # Use same pretrained model
            self.xception_model = xception(weights=None)
            num_features = self.xception_model.fc.in_features
            self.xception_model.fc = nn.Linear(num_features, 2)
            
            if xception_weights_path.exists():
                try:
                    checkpoint = torch.load(xception_weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.xception_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.xception_model.load_state_dict(checkpoint)
                    logger.info(f"Pretrained Xception model loaded from {xception_weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load pretrained Xception weights: {e}. Using ImageNet pretrained as fallback.")
                    pretrained = xception(weights=Xception_Weights.IMAGENET1K_V1)
                    self.xception_model = pretrained
                    # Replace final layer
                    num_features = self.xception_model.fc.in_features
                    self.xception_model.fc = nn.Linear(num_features, 2)
            else:
                logger.warning(f"Pretrained Xception model not found. Using ImageNet pretrained as fallback.")
                pretrained = xception(weights=Xception_Weights.IMAGENET1K_V1)
                self.xception_model = pretrained
                num_features = self.xception_model.fc.in_features
                self.xception_model.fc = nn.Linear(num_features, 2)
            
            self.xception_model.to(self.device)
            self.xception_model.eval()
            
            # Load EfficientNet-B7 (using same pretrained model)
            efficientnet_weights_path = ensemble_weights_path  # Use same pretrained model
            self.efficientnet_model = efficientnet_b7(weights=None)
            num_features = self.efficientnet_model.classifier[1].in_features
            self.efficientnet_model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 2)
            )
            
            if efficientnet_weights_path.exists():
                try:
                    checkpoint = torch.load(efficientnet_weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.efficientnet_model.load_state_dict(checkpoint)
                    logger.info(f"Pretrained EfficientNet-B7 model loaded from {efficientnet_weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load pretrained EfficientNet-B7 weights: {e}. Using ImageNet pretrained as fallback.")
                    pretrained = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
                    self.efficientnet_model.features = pretrained.features
            else:
                logger.warning(f"Pretrained EfficientNet-B7 model not found. Using ImageNet pretrained as fallback.")
                pretrained = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
                self.efficientnet_model.features = pretrained.features
            
            self.efficientnet_model.to(self.device)
            self.efficientnet_model.eval()
            
            self._loaded = True
            logger.info(f"Ensemble detector loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble models: {e}")
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        results = []
        for (x, y, w, h) in faces:
            margin = int(w * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'detection_confidence': 0.9,
                'crop': image[y1:y2, x1:x2].copy()
            })
        return results
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run ensemble prediction on image."""
        if not self._loaded:
            return {'error': 'Models not loaded', 'fake_probability': 0.5}
        
        try:
            # Preprocess for both models
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            xception_input = self.xception_transform(image_bgr)
            efficientnet_input = self.efficientnet_transform(image_bgr)
            
            xception_input = xception_input.unsqueeze(0).to(self.device)
            efficientnet_input = efficientnet_input.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Xception prediction
                xception_outputs = self.xception_model(xception_input)
                xception_probs = torch.nn.functional.softmax(xception_outputs, dim=1)[0]
                xception_fake_prob = float(xception_probs[1].cpu().numpy())
                
                # EfficientNet-B7 prediction
                efficientnet_outputs = self.efficientnet_model(efficientnet_input)
                efficientnet_probs = torch.nn.functional.softmax(efficientnet_outputs, dim=1)[0]
                efficientnet_fake_prob = float(efficientnet_probs[1].cpu().numpy())
            
            # Weighted ensemble
            ensemble_fake_prob = (
                self.xception_weight * xception_fake_prob +
                self.efficientnet_weight * efficientnet_fake_prob
            )
            
            return {
                'fake_probability': ensemble_fake_prob,
                'real_probability': 1.0 - ensemble_fake_prob,
                'xception': {
                    'fake_probability': xception_fake_prob,
                    'real_probability': 1.0 - xception_fake_prob
                },
                'efficientnet_b7': {
                    'fake_probability': efficientnet_fake_prob,
                    'real_probability': 1.0 - efficientnet_fake_prob
                },
                'ensemble_weights': {
                    'xception': self.xception_weight,
                    'efficientnet_b7': self.efficientnet_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'error': str(e), 'fake_probability': 0.5}
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """Full ensemble analysis."""
        result = {
            'overall_prediction': None,
            'face_predictions': [],
            'faces_detected': 0,
            'analysis_type': 'full_image',
            'model': 'Ensemble-Xception-EfficientNet-B7 (Pretrained)'
        }
        
        if not self._loaded:
            if not self.load_model():
                result['error'] = 'Models not loaded'
                result['overall_prediction'] = {'fake_probability': 0.5}
                return result
        
        faces = self.detect_faces(image)
        result['faces_detected'] = len(faces)
        
        if len(faces) == 0:
            prediction = self.predict(image)
            result['overall_prediction'] = prediction
            result['analysis_type'] = 'full_image'
        else:
            face_probs = []
            for i, face in enumerate(faces):
                if face['crop'].size > 0:
                    prediction = self.predict(face['crop'])
                    face_probs.append(prediction.get('fake_probability', 0.5))
                    result['face_predictions'].append({
                        'face_id': i,
                        'bbox': face['bbox'],
                        'detection_confidence': face['detection_confidence'],
                        **prediction
                    })
            
            if face_probs:
                max_fake_prob = max(face_probs)
                result['overall_prediction'] = {
                    'fake_probability': max_fake_prob,
                    'real_probability': 1 - max_fake_prob,
                    'aggregation': 'max'
                }
            else:
                prediction = self.predict(image)
                result['overall_prediction'] = prediction
        
        return result


# Singleton instance
ensemble_detector = EnsembleDetector()
