/**
 * API Service for Deepfake Detection
 */
import axios from 'axios';

const API_BASE = '/api/v1/analyze';

// Types
export interface FaceDetection {
    face_id: number;
    bbox: number[];
    detection_confidence: number;
    fake_probability?: number;
}

export interface Explanation {
    summary: string;
    factors: string[];
    recommendation: string;
}

export interface ImageAnalysisResult {
    analysis_id: string;
    timestamp: string;
    media_type: string;
    filename: string;
    classification: 'AUTHENTIC' | 'SUSPICIOUS' | 'MANIPULATED';
    confidence: 'LOW' | 'MEDIUM' | 'HIGH';
    risk_score: number;
    signals: Record<string, any>;
    face_detections: FaceDetection[];
    explanation: Explanation;
    processing_time_ms: number;
}

export interface FrameAnalysis {
    frame_index: number;
    timestamp_seconds: number;
    risk_score: number;
    classification: string;
    faces_detected: number;
}

export interface VideoAnalysisResult {
    analysis_id: string;
    timestamp: string;
    media_type: string;
    filename: string;
    video_info: {
        duration_seconds: number;
        total_frames: number;
        fps: number;
        resolution: string;
        frames_analyzed: number;
    };
    classification: 'AUTHENTIC' | 'SUSPICIOUS' | 'MANIPULATED';
    confidence: 'LOW' | 'MEDIUM' | 'HIGH';
    risk_score: number;
    average_risk_score: number;
    frame_analysis: FrameAnalysis[];
    temporal_consistency?: string;
    audio_analysis?: any;
    explanation: Explanation;
    processing_time_ms: number;
}

export interface AudioAnalysisResult {
    analysis_id: string;
    timestamp: string;
    media_type: string;
    filename: string;
    audio_info: {
        duration_seconds: number;
        sample_rate: number;
    };
    classification: 'AUTHENTIC' | 'SUSPICIOUS' | 'SYNTHETIC';
    confidence: 'LOW' | 'MEDIUM' | 'HIGH';
    risk_score: number;
    prediction: Record<string, number>;
    audio_features: Record<string, number>;
    processing_time_ms: number;
}

export type AnalysisResult = ImageAnalysisResult | VideoAnalysisResult | AudioAnalysisResult;

// API Functions
export async function analyzeImage(file: File): Promise<ImageAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post<ImageAnalysisResult>(`${API_BASE}/image/`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
}

export async function analyzeVideo(file: File, numFrames: number = 16): Promise<VideoAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post<VideoAnalysisResult>(
        `${API_BASE}/video/?num_frames=${numFrames}`,
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        }
    );

    return response.data;
}

export async function analyzeAudio(file: File): Promise<AudioAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post<AudioAnalysisResult>(`${API_BASE}/audio/`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
}

export async function checkHealth(): Promise<any> {
    const response = await axios.get('/health');
    return response.data;
}

// Helper to detect media type from file
export function getMediaType(file: File): 'image' | 'video' | 'audio' | 'unknown' {
    const imageTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    const videoTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska'];
    const audioTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/flac', 'audio/ogg'];

    if (imageTypes.includes(file.type)) return 'image';
    if (videoTypes.includes(file.type)) return 'video';
    if (audioTypes.includes(file.type)) return 'audio';

    // Fallback to extension
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    if (['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext)) return 'image';
    if (['mp4', 'avi', 'mov', 'webm', 'mkv'].includes(ext)) return 'video';
    if (['wav', 'mp3', 'm4a', 'flac', 'ogg'].includes(ext)) return 'audio';

    return 'unknown';
}

// Helper to analyze any media file
export async function analyzeMedia(file: File): Promise<AnalysisResult> {
    const mediaType = getMediaType(file);

    switch (mediaType) {
        case 'image':
            return analyzeImage(file);
        case 'video':
            return analyzeVideo(file);
        case 'audio':
            return analyzeAudio(file);
        default:
            throw new Error(`Unsupported file type: ${file.type || file.name}`);
    }
}

// ============ ADVANCED ANALYSIS ============

export async function analyzeImageAdvanced(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/advanced/image/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
}

export async function analyzeAudioAdvanced(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/advanced/audio/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
}

export async function getAdvancedStatus(): Promise<any> {
    const response = await axios.get(`${API_BASE}/advanced/status`);
    return response.data;
}

export async function analyzeVideoAdvanced(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE}/advanced/video/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
}

export async function analyzeMediaAdvanced(file: File): Promise<any> {
    const mediaType = getMediaType(file);
    if (mediaType === 'image') return analyzeImageAdvanced(file);
    if (mediaType === 'video') return analyzeVideoAdvanced(file);
    // Audio uses basic analysis (enhanced not available)
    if (mediaType === 'audio') return analyzeAudio(file);
    return analyzeVideo(file);
}

// PDF Report Download
export async function downloadPDFReport(analysisResult: any): Promise<void> {
    try {
        const response = await axios.post(
            `${API_BASE}/report/from-json`,
            analysisResult,
            { responseType: 'blob' }
        );

        // Create download link
        const blob = new Blob([response.data], { type: 'application/pdf' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;

        // Generate filename
        const filename = analysisResult.filename || 'analysis';
        const safeName = filename.replace(/[^a-zA-Z0-9]/g, '_');
        link.download = `deepfake_report_${safeName}_${Date.now()}.pdf`;

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Failed to download PDF:', error);
        throw error;
    }
}
