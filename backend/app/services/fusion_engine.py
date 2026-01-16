"""Fusion Engine - Combines signals from multiple detectors."""
from typing import Dict, Any, Optional, List
import numpy as np
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class FusionEngine:
    """Fuses signals from multiple detection pathways into final decision."""
    
    def __init__(self):
        self._initialized = False
        
        # Risk thresholds
        self.HIGH_RISK_THRESHOLD = 70
        self.MEDIUM_RISK_THRESHOLD = 40
        
    def initialize(self):
        """Initialize the fusion engine."""
        self._initialized = True
        logger.info("Fusion engine initialized")
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def compute_risk_score(
        self, 
        signals: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """
        Compute weighted risk score from multiple signals.
        
        Args:
            signals: Dict of signal_name -> probability (0-1)
            weights: Dict of signal_name -> weight
            
        Returns:
            Risk score from 0-100
        """
        if not signals:
            return 50.0  # Neutral if no signals
        
        # Filter to only signals we have weights for
        valid_signals = {k: v for k, v in signals.items() if k in weights}
        
        if not valid_signals:
            return 50.0
        
        # Normalize weights for available signals
        total_weight = sum(weights[k] for k in valid_signals)
        
        if total_weight == 0:
            return 50.0
        
        # Weighted average
        weighted_sum = sum(
            valid_signals[k] * weights[k] for k in valid_signals
        )
        
        return (weighted_sum / total_weight) * 100
    
    def classify(self, risk_score: float) -> str:
        """
        Classify based on risk score.
        
        Returns:
            'AUTHENTIC', 'SUSPICIOUS', or 'MANIPULATED'
        """
        if risk_score >= self.HIGH_RISK_THRESHOLD:
            return 'MANIPULATED'
        elif risk_score >= self.MEDIUM_RISK_THRESHOLD:
            return 'SUSPICIOUS'
        else:
            return 'AUTHENTIC'
    
    def compute_confidence(
        self, 
        risk_score: float, 
        signal_agreement: float
    ) -> str:
        """
        Determine confidence level based on risk score and signal agreement.
        
        Args:
            risk_score: 0-100
            signal_agreement: 0-1 (1 = all signals agree)
            
        Returns:
            'LOW', 'MEDIUM', or 'HIGH'
        """
        # Distance from uncertainty (50 is most uncertain)
        distance_from_uncertain = abs(risk_score - 50)
        
        if distance_from_uncertain > 35 and signal_agreement > 0.7:
            return 'HIGH'
        elif distance_from_uncertain > 15 or signal_agreement > 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def compute_signal_agreement(self, signals: Dict[str, float]) -> float:
        """
        Compute how much the signals agree with each other.
        
        Returns:
            Agreement score from 0-1 (1 = perfect agreement)
        """
        if len(signals) < 2:
            return 0.5  # Can't compute agreement with 1 signal
        
        values = list(signals.values())
        
        # Agreement = 1 - standard deviation of signals
        # If all signals are the same, std=0, agreement=1
        std = np.std(values)
        agreement = max(0, 1 - std)
        
        return float(agreement)
    
    def fuse_image_signals(
        self,
        visual_result: Dict[str, Any],
        forensic_result: Dict[str, Any],
        metadata_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuse signals from image analysis.
        """
        signals = {}
        weights = {}
        
        # Ensemble Strategy: Visual (0.6) + Forensic (0.4)
        
        # Visual signal (Xception)
        has_visual = False
        if visual_result and visual_result.get('overall_prediction'):
            pred = visual_result['overall_prediction']
            if 'fake_probability' in pred:
                signals['visual'] = pred['fake_probability']
                weights['visual'] = 0.6
                has_visual = True
        
        # Forensic signal
        has_forensic = False
        if forensic_result and forensic_result.get('prediction'):
            pred = forensic_result['prediction']
            if 'fake_probability' in pred:
                signals['forensic'] = pred['fake_probability']
                weights['forensic'] = 0.4 if has_visual else 1.0
                has_forensic = True
                
        # Fallback if visual failed
        if has_forensic and not has_visual:
            weights['forensic'] = 1.0
        
        # Compute final scores
        risk_score = self.compute_risk_score(signals, weights)
        signal_agreement = self.compute_signal_agreement(signals)
        
        return {
            'risk_score': round(risk_score, 2),
            'classification': self.classify(risk_score),
            'confidence': self.compute_confidence(risk_score, signal_agreement),
            'signal_agreement': round(signal_agreement, 3),
            'signals_used': list(signals.keys()),
            'signal_values': signals,
            'weights_used': {k: weights[k] for k in signals}
        }
    
    def fuse_video_signals(
        self,
        frame_results: List[Dict[str, Any]],
        temporal_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuse signals from video analysis.
        """
        signals = {}
        weights = {}
        
        # Video Strategy: Temporal (0.4) + Visual Frames (0.4) + Forensic (not separate here but part of frames?)
        # Actually frame_results usually contain visual+forensic fused per frame
        
        # Aggregate frame-level scores (Spatial Analysis)
        if frame_results:
            frame_scores = [
                f.get('risk_score', 50) / 100 
                for f in frame_results 
                if 'risk_score' in f
            ]
            if frame_scores:
                # Use mean score across frames for stability
                signals['spatial'] = sum(frame_scores) / len(frame_scores)
                weights['spatial'] = 0.3
        
        # Temporal signal (Main Detector)
        if temporal_result and temporal_result.get('prediction'):
            pred = temporal_result['prediction']
            # Handles both existing formats just in case
            if 'fake_probability' in pred:
                signals['temporal'] = pred['fake_probability']
                weights['temporal'] = 0.5 # High weight for temporal model
            elif 'inconsistency_score' in pred:
                signals['temporal'] = pred['inconsistency_score']
                weights['temporal'] = 0.5
        
        # Audio signal
        if audio_result and audio_result.get('prediction'):
            pred = audio_result['prediction']
            if 'synthetic_probability' in pred:
                signals['audio'] = pred['synthetic_probability']
                weights['audio'] = 0.2
        
        risk_score = self.compute_risk_score(signals, weights)
        signal_agreement = self.compute_signal_agreement(signals)
        
        return {
            'risk_score': round(risk_score, 2),
            'classification': self.classify(risk_score),
            'confidence': self.compute_confidence(risk_score, signal_agreement),
            'signal_agreement': round(signal_agreement, 3),
            'signals_used': list(signals.keys()),
            'frames_analyzed': len(frame_results) if frame_results else 0
        }


# Singleton instance
fusion_engine = FusionEngine()
