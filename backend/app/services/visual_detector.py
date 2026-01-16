
import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class VisualDetector:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "xception_deepfake_detector.onnx")
        self.face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Visual model not found at {self.model_path}")
                return

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Visual Xception detector loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load visual detector: {e}")
            self.session = None

    def load_model(self) -> bool:
        """Public method to load model (called by main.py on startup)."""
        if self.session is None:
            self._load_model()
        return self.session is not None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None

    def preprocess(self, face_img):
        """Preprocess face image for Xception model. Input is RGB."""
        # Resize to 299x299 (Standard Xception)
        img = cv2.resize(face_img, (299, 299))
        
        # Input is already RGB from API, no conversion needed
        # Normalize: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # Transpose to (C, H, W) for PyTorch-trained ONNX
        img = img.transpose(2, 0, 1)
        
        # Batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def analyze(self, image):
        if self.session is None:
            return {"error": "Visual model not loaded", "risk_score": 0, "classification": "UNKNOWN"}

        try:
            # Detect Face
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Input image is RGB from api
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # Fallback: Use center crop or entire image if no face
                logger.warning("No face detected, using full image")
                face = image
                face_coords = [0, 0, image.shape[1], image.shape[0]]
                faces_detected = 0
            else:
                faces_detected = len(faces)
                # Select largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add margin (simulating the python code's 20px margin logic)
                # Note: code had 20px fixed margin. Relative margin might be safer but let's stick to logic.
                margin = 20
                h_img, w_img = image.shape[:2]
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)
                
                face = image[y1:y2, x1:x2]
                face_coords = [x1, y1, x2, y2]

            # Preprocess
            input_tensor = self.preprocess(face)
            
            # Inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            logits = outputs[0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Class 0: Real, Class 1: Fake (Usually, checking based on user code provided)
            # User code: fake_prob = probs[0, 1].item()
            fake_prob = float(probs[0][1])
            real_prob = float(probs[0][0])
            
            risk_score = fake_prob * 100
            
            prediction = "MANIPULATED" if fake_prob > 0.55 else "AUTHENTIC" # Slight buffer for robustness
            
            return {
                "analysis_type": "xception_visual",
                "risk_score": risk_score,
                "confidence": "HIGH" if abs(risk_score - 50) > 30 else "LOW",
                "classification": prediction,
                "faces_detected": faces_detected,
                "overall_prediction": {
                     "fake_probability": fake_prob,
                     "real_probability": real_prob
                },
                "face_predictions": [{
                    "face_id": 0,
                    "bbox": face_coords,
                    "fake_probability": fake_prob,
                    "detection_confidence": 0.9
                }]
            }

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return {"error": str(e), "risk_score": 0, "classification": "ERROR"}

# Singleton
visual_detector = VisualDetector()
