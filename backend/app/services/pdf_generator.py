"""PDF Report Generation Service for Deepfake Detection Results."""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from datetime import datetime
from typing import Dict, Any
import io
import os
import logging

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate detailed PDF analysis reports."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e'),
            alignment=1  # Center
        ))
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            alignment=1
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#333333'),
            borderWidth=0,
            borderPadding=5
        ))
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        ))
    
    def _get_classification_color(self, classification: str) -> colors.Color:
        """Get color based on classification."""
        if classification == 'AUTHENTIC':
            return colors.HexColor('#22c55e')  # Green
        elif classification == 'SUSPICIOUS':
            return colors.HexColor('#f59e0b')  # Orange
        else:  # MANIPULATED/SYNTHETIC
            return colors.HexColor('#ef4444')  # Red
    
    def _create_risk_gauge(self, risk_score: float) -> Drawing:
        """Create a visual risk gauge."""
        d = Drawing(200, 100)
        
        # Background bar
        d.add(Rect(0, 40, 200, 20, fillColor=colors.HexColor('#e5e5e5'), strokeColor=None))
        
        # Risk level bar
        color = self._get_classification_color(
            'AUTHENTIC' if risk_score < 40 else ('SUSPICIOUS' if risk_score < 70 else 'MANIPULATED')
        )
        d.add(Rect(0, 40, risk_score * 2, 20, fillColor=color, strokeColor=None))
        
        # Risk score text
        d.add(String(100, 70, f"{risk_score:.1f}%", fontSize=14, textAnchor='middle'))
        d.add(String(100, 20, "Risk Score", fontSize=10, textAnchor='middle', fillColor=colors.gray))
        
        return d
    
    def generate_image_report(self, analysis_result: Dict[str, Any], output_path: str = None) -> bytes:
        """
        Generate a detailed PDF report for image analysis.
        
        Args:
            analysis_result: The analysis result dictionary
            output_path: Optional file path to save the PDF
            
        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        story = []
        
        # Title
        story.append(Paragraph("Deepfake Detection Analysis Report", self.styles['ReportTitle']))
        story.append(Paragraph("Media Authenticity Verification", self.styles['ReportSubtitle']))
        story.append(Spacer(1, 30))
        
        # Classification Banner
        classification = analysis_result.get('classification', 'UNKNOWN')
        confidence = analysis_result.get('confidence', 'MEDIUM')
        risk_score = analysis_result.get('risk_score', 0)
        
        class_color = self._get_classification_color(classification)
        
        classification_data = [[
            Paragraph(f"<b>Classification: {classification}</b>", 
                     ParagraphStyle('ClassBanner', fontSize=16, textColor=colors.white, alignment=1))
        ]]
        classification_table = Table(classification_data, colWidths=[400])
        classification_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), class_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ]))
        story.append(classification_table)
        story.append(Spacer(1, 20))
        
        # Analysis Summary Section
        story.append(Paragraph("Analysis Summary", self.styles['SectionHeader']))
        
        analysis_id = analysis_result.get('analysis_id', 'N/A')
        timestamp = analysis_result.get('timestamp', datetime.now().isoformat())
        filename = analysis_result.get('filename', 'Unknown')
        model = analysis_result.get('model', 'DeepVision-v2')
        processing_time = analysis_result.get('processing_time_ms', 0)
        
        # Parse timestamp if string
        if isinstance(timestamp, str):
            try:
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = ts.strftime("%B %d, %Y at %I:%M %p")
            except:
                timestamp_str = timestamp
        else:
            timestamp_str = str(timestamp)
        
        summary_data = [
            ['Analysis ID:', analysis_id],
            ['Date & Time:', timestamp_str],
            ['Filename:', filename],
            ['Media Type:', analysis_result.get('media_type', 'image').upper()],
            ['Detection Model:', model],
            ['Processing Time:', f"{processing_time} ms"],
        ]
        
        summary_table = Table(summary_data, colWidths=[150, 280])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Detection Results Section
        story.append(Paragraph("Detection Results", self.styles['SectionHeader']))
        
        prediction = analysis_result.get('prediction', {})
        fake_prob = prediction.get('fake_probability', 0) * 100
        real_prob = prediction.get('real_probability', 0) * 100
        
        results_data = [
            ['Risk Score:', f"{risk_score:.2f}%"],
            ['Confidence Level:', confidence],
            ['Authenticity Probability:', f"{real_prob:.2f}%"],
            ['Manipulation Probability:', f"{fake_prob:.2f}%"],
        ]
        
        results_table = Table(results_data, colWidths=[200, 230])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Interpretation Section
        story.append(Paragraph("Interpretation", self.styles['SectionHeader']))
        
        if classification == 'AUTHENTIC':
            interpretation = f"""
            The analysis indicates that this media is <b>likely authentic</b> with a {real_prob:.1f}% authenticity probability.
            The risk score of {risk_score:.1f}% falls within the safe range, suggesting minimal signs of manipulation.
            <br/><br/>
            <b>Key Findings:</b><br/>
            • No significant deepfake artifacts detected<br/>
            • Visual patterns consistent with authentic media<br/>
            • Image forensics show natural characteristics<br/>
            """
        elif classification == 'SUSPICIOUS':
            interpretation = f"""
            The analysis has flagged this media as <b>suspicious</b> with a {fake_prob:.1f}% manipulation probability.
            The risk score of {risk_score:.1f}% indicates potential signs of manipulation that warrant further investigation.
            <br/><br/>
            <b>Key Findings:</b><br/>
            • Some anomalies detected in visual patterns<br/>
            • Borderline forensic indicators present<br/>
            • Recommend additional verification<br/>
            """
        else:  # MANIPULATED
            interpretation = f"""
            The analysis strongly suggests this media has been <b>manipulated</b> with a {fake_prob:.1f}% manipulation probability.
            The high risk score of {risk_score:.1f}% indicates significant signs of deepfake or synthetic generation.
            <br/><br/>
            <b>Key Findings:</b><br/>
            • Strong deepfake artifacts detected<br/>
            • Inconsistent visual patterns identified<br/>
            • Forensic analysis reveals manipulation signatures<br/>
            """
        
        story.append(Paragraph(interpretation, self.styles['ReportBody']))
        story.append(Spacer(1, 20))
        
        # Confidence Explanation
        story.append(Paragraph("Confidence Level Explanation", self.styles['SectionHeader']))
        
        if confidence == 'HIGH':
            conf_text = """
            <b>HIGH Confidence:</b> The detection models show strong agreement with probability scores above 80-90%.
            This result can be considered highly reliable for decision-making purposes.
            """
        elif confidence == 'MEDIUM':
            conf_text = """
            <b>MEDIUM Confidence:</b> The detection shows moderate certainty. While the result is indicative,
            additional verification may be beneficial for critical decisions.
            """
        else:
            conf_text = """
            <b>LOW Confidence:</b> The detection shows some uncertainty. The probability scores are close to the
            threshold boundaries. Further analysis with additional tools is recommended.
            """
        
        story.append(Paragraph(conf_text, self.styles['ReportBody']))
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(Paragraph("─" * 60, self.styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
            ParagraphStyle('Footer', fontSize=9, textColor=colors.gray, alignment=1)
        ))
        story.append(Paragraph(
            "<i>Deepfake Detection & Media Authenticity Analyzer</i>",
            ParagraphStyle('Footer2', fontSize=9, textColor=colors.gray, alignment=1)
        ))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            logger.info(f"PDF report saved to {output_path}")
        
        return pdf_bytes


# Singleton
pdf_generator = PDFReportGenerator()
