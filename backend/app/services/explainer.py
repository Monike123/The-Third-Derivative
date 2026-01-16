"""Explainer Service - Generate human-readable explanations."""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ExplainerService:
    """Generate human-readable explanations for detection results."""
    
    def explain_image_result(
        self,
        visual_result: Dict[str, Any],
        forensic_result: Dict[str, Any],
        fused_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation for image analysis results.
        
        Returns:
            Dict with 'summary', 'factors', and 'recommendation'
        """
        factors = []
        
        # Analyze visual signals
        if visual_result.get('overall_prediction'):
            visual_prob = visual_result['overall_prediction'].get('fake_probability', 0.5)
            
            if visual_prob > 0.7:
                factors.append("Visual patterns strongly suggest manipulation")
            elif visual_prob > 0.5:
                factors.append("Some visual anomalies detected")
            elif visual_prob > 0.3:
                factors.append("Minor visual irregularities found")
            else:
                factors.append("Visual patterns appear natural")
        
        # Analyze forensic signals
        if forensic_result.get('prediction'):
            forensic_prob = forensic_result['prediction'].get('fake_probability', 0.5)
            features = forensic_result.get('features', {})
            
            if forensic_prob > 0.7:
                factors.append("Forensic analysis indicates likely manipulation")
                
                # Explain specific forensic findings
                if features.get('high_freq_ratio', 0.5) < 0.3:
                    factors.append("Unusual frequency distribution (possible synthetic generation)")
                
                if features.get('noise_var_ratio', 1) > 2:
                    factors.append("Inconsistent noise patterns across color channels")
                
                if features.get('sharpness', 0) < 100:
                    factors.append("Unusually low image sharpness detected")
                    
            elif forensic_prob > 0.5:
                factors.append("Some forensic anomalies detected")
            else:
                factors.append("Forensic patterns consistent with authentic media")
        
        # Face detection context
        faces_detected = visual_result.get('faces_detected', 0)
        if faces_detected > 0:
            factors.append(f"{faces_detected} face(s) analyzed for manipulation signs")
        else:
            factors.append("Full image analyzed (no faces detected)")
        
        # Generate summary
        classification = fused_result.get('classification', 'UNKNOWN')
        risk_score = fused_result.get('risk_score', 50)
        
        if classification == 'MANIPULATED':
            summary = (
                f"This media shows strong signs of manipulation with a risk score of "
                f"{risk_score:.0f}/100. Multiple detection signals indicate the content "
                f"has likely been altered or synthetically generated."
            )
        elif classification == 'SUSPICIOUS':
            summary = (
                f"This media shows some signs that warrant caution with a risk score of "
                f"{risk_score:.0f}/100. Some anomalies were detected but are not conclusive."
            )
        else:
            summary = (
                f"This media appears authentic with a risk score of {risk_score:.0f}/100. "
                f"Detection signals are consistent with unaltered content."
            )
        
        return {
            'summary': summary,
            'factors': factors,
            'recommendation': self._get_recommendation(classification)
        }
    
    def explain_video_result(
        self,
        frame_count: int,
        fused_result: Dict[str, Any],
        audio_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate explanation for video analysis results."""
        factors = []
        
        classification = fused_result.get('classification', 'UNKNOWN')
        risk_score = fused_result.get('risk_score', 50)
        
        factors.append(f"{frame_count} frames analyzed for manipulation")
        
        if fused_result.get('signals_used'):
            signals = fused_result['signals_used']
            factors.append(f"Detection signals used: {', '.join(signals)}")
        
        if audio_result and audio_result.get('prediction'):
            audio_prob = audio_result['prediction'].get('synthetic_probability', 0.5)
            if audio_prob > 0.6:
                factors.append("Audio track shows signs of synthetic generation")
            else:
                factors.append("Audio track appears authentic")
        
        if classification == 'MANIPULATED':
            summary = (
                f"This video shows strong signs of manipulation with a risk score of "
                f"{risk_score:.0f}/100. Analysis detected potential deepfake artifacts."
            )
        elif classification == 'SUSPICIOUS':
            summary = (
                f"This video warrants caution with a risk score of {risk_score:.0f}/100. "
                f"Some frames show anomalies that may indicate editing."
            )
        else:
            summary = (
                f"This video appears authentic with a risk score of {risk_score:.0f}/100."
            )
        
        return {
            'summary': summary,
            'factors': factors,
            'recommendation': self._get_recommendation(classification)
        }
    
    def _get_recommendation(self, classification: str) -> str:
        """Get action recommendation based on classification."""
        recommendations = {
            'MANIPULATED': (
                "Exercise extreme caution. Consider this media potentially misleading "
                "until verified through independent sources. Do not share without verification."
            ),
            'SUSPICIOUS': (
                "Treat with caution. Seek additional verification before sharing or "
                "acting on this content. Cross-reference with trusted sources."
            ),
            'AUTHENTIC': (
                "This media passes our authenticity checks, but always verify important "
                "information through multiple sources."
            )
        }
        return recommendations.get(classification, "Unable to provide recommendation.")


# Singleton instance
explainer = ExplainerService()
