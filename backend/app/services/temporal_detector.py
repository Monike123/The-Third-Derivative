
import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class TemporalDeepfakeDetector:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "temporal_deepfake_detector.onnx")
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Temporal model not found at {self.model_path}")
                return

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Temporal detector loaded from {self.model_path}")
            logger.info(f"Input shape: {self.session.get_inputs()[0].shape}")
            
        except Exception as e:
            logger.error(f"Failed to load temporal detector: {e}")
            self.session = None

    def video_to_clip(self, video_path, num_frames=8, size=112):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniform sampling if video is long enough
        if total_frames >= num_frames:
             indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
             for idx in indices:
                 cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                 ret, frame = cap.read()
                 if ret:
                     frame = cv2.resize(frame, (size, size))
                     # BGR to RGB
                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     frames.append(frame)
        else:
            # Read all frames -> loop/pad if needed (simple read loop)
            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (size, size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()

        # Handle short videos by repeating last frame
        while len(frames) < num_frames and len(frames) > 0:
            frames.append(frames[-1])
            
        if len(frames) < num_frames:
             raise ValueError("Video too short for inference")

        # Convert to numpy and normalize
        clip = np.array(frames[:num_frames], dtype=np.float32) / 255.0
        
        # (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
        # Note: Original code transpose was (0, 3, 1, 2) which implies input frames list is (T, H, W, C)? 
        # Yes, cv2 reads (H, W, C).
        # But we need to be careful. Code said: clip = clip.transpose(0, 3, 1, 2)   # (T,C,H,W)
        # Verify: np.array(frames) shape is (T, H, W, C).
        # (0, 3, 1, 2) -> T, C, H, W. Correct.
        clip = clip.transpose(0, 3, 1, 2)
        clip = np.expand_dims(clip, axis=0) # (B, T, C, H, W)

        return clip

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum(axis=1, keepdims=True)

    def analyze(self, video_path):
        if self.session is None:
            return {"error": "Temporal model not loaded"}

        try:
            clip = self.video_to_clip(video_path, num_frames=8)
            
            logits = self.session.run(
                [self.output_name],
                {self.input_name: clip}
            )[0]

            probs = self.softmax(logits)
            real_conf = float(probs[0][0])
            fake_conf = float(probs[0][1])
            
            prediction = "MANIPULATED" if fake_conf > real_conf else "AUTHENTIC"
            risk_score = fake_conf * 100

            return {
                "classification": prediction,
                "risk_score": risk_score,
                "confidence": "HIGH" if abs(risk_score - 50) > 30 else "LOW",
                "prediction": {
                    "real_probability": real_conf,
                    "fake_probability": fake_conf
                }
            }

        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {"error": str(e)}

# Singleton
temporal_detector = TemporalDeepfakeDetector()
