import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Upload, Image, Video, Music, AlertTriangle, CheckCircle,
    XCircle, Loader, FileText, ChevronDown, ChevronUp, Zap, Shield, Download
} from 'lucide-react';
import { analyzeMedia, analyzeMediaAdvanced, downloadPDFReport, getMediaType } from '../services/api';
import './AnalyzePage.css';

export function AnalyzePage() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [showDetails, setShowDetails] = useState(false);
    const [mode, setMode] = useState<'basic' | 'advanced'>('basic');


    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (file) {
            setFile(file);
            setError(null);
            setResult(null);

            // Create preview for images
            const mediaType = getMediaType(file);
            if (mediaType === 'image') {
                const reader = new FileReader();
                reader.onload = () => setPreview(reader.result as string);
                reader.readAsDataURL(file);
            } else {
                setPreview(null);
            }
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
            'video/*': ['.mp4', '.avi', '.mov', '.webm', '.mkv'],
            'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        },
        maxSize: 100 * 1024 * 1024, // 100MB
        multiple: false
    });

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const analysisResult = mode === 'advanced'
                ? await analyzeMediaAdvanced(file)
                : await analyzeMedia(file);
            setResult(analysisResult);
        } catch (err: any) {
            const detail = err.response?.data?.detail || err.message || 'Analysis failed';
            setError(mode === 'advanced' && (detail.includes('HF_TOKEN') || detail.includes('503'))
                ? 'Enhanced detection service temporarily unavailable'
                : detail);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const [downloading, setDownloading] = useState(false);

    const handleDownloadReport = async () => {
        if (!result) return;
        setDownloading(true);
        try {
            await downloadPDFReport(result);
        } catch (err) {
            setError('Failed to download report');
        } finally {
            setDownloading(false);
        }
    };

    // Get interpretation based on classification
    const getInterpretation = () => {
        if (!result) return null;
        const classification = result.classification;
        const riskScore = result.risk_score || 0;
        const prediction = result.prediction || {};
        const realProb = (prediction.real_probability || 0) * 100;
        const fakeProb = (prediction.fake_probability || prediction.synthetic_probability || 0) * 100;

        if (classification === 'AUTHENTIC') {
            return {
                title: 'Media Appears Authentic',
                description: `This media is likely authentic with ${realProb.toFixed(1)}% authenticity probability. The risk score of ${riskScore.toFixed(1)}% falls within the safe range.`,
                findings: [
                    'No significant deepfake artifacts detected',
                    'Visual patterns consistent with authentic media',
                    'Forensic analysis shows natural characteristics'
                ],
                recommendation: 'This media can be considered genuine based on our analysis.'
            };
        } else if (classification === 'SUSPICIOUS') {
            return {
                title: 'Media Flagged as Suspicious',
                description: `This media shows potential signs of manipulation with ${fakeProb.toFixed(1)}% manipulation probability. Further investigation is recommended.`,
                findings: [
                    'Some anomalies detected in visual patterns',
                    'Borderline forensic indicators present',
                    'Results require additional verification'
                ],
                recommendation: 'Consider using additional verification methods before making critical decisions.'
            };
        } else {
            return {
                title: 'Media Likely Manipulated',
                description: `Strong indicators of manipulation detected with ${fakeProb.toFixed(1)}% manipulation probability. Risk score: ${riskScore.toFixed(1)}%.`,
                findings: [
                    'Strong deepfake artifacts detected',
                    'Inconsistent visual patterns identified',
                    'Forensic analysis reveals manipulation signatures'
                ],
                recommendation: 'This media should be treated with caution and not used as authentic evidence.'
            };
        }
    };

    const getMediaIcon = () => {
        if (!file) return <Upload size={48} />;
        const type = getMediaType(file);
        switch (type) {
            case 'image': return <Image size={48} />;
            case 'video': return <Video size={48} />;
            case 'audio': return <Music size={48} />;
            default: return <FileText size={48} />;
        }
    };

    const getClassificationBadge = (classification: string) => {
        switch (classification) {
            case 'AUTHENTIC':
                return <span className="badge badge-success"><CheckCircle size={14} /> Authentic</span>;
            case 'SUSPICIOUS':
                return <span className="badge badge-warning"><AlertTriangle size={14} /> Suspicious</span>;
            case 'MANIPULATED':
            case 'SYNTHETIC':
                return <span className="badge badge-danger"><XCircle size={14} /> Manipulated</span>;
            default:
                return <span className="badge">{classification}</span>;
        }
    };

    const getRiskColor = (score: number) => {
        if (score < 40) return 'var(--success)';
        if (score < 70) return 'var(--warning)';
        return 'var(--danger)';
    };

    return (
        <div className="analyze-page">
            <div className="container">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="page-header"
                >
                    <h1>Analyze Media</h1>
                    <p>Upload an image, video, or audio file to check for manipulation</p>

                    {/* Mode Toggle */}
                    <div className="mode-toggle">
                        <button
                            className={`mode-btn ${mode === 'basic' ? 'active' : ''}`}
                            onClick={() => setMode('basic')}
                        >
                            <Shield size={18} />
                            Standard Analysis
                        </button>
                        <button
                            className={`mode-btn ${mode === 'advanced' ? 'active' : ''}`}
                            onClick={() => setMode('advanced')}
                        >
                            <Zap size={18} />
                            Enhanced Detection
                        </button>
                    </div>
                    {mode === 'advanced' && (
                        <p className="mode-note">Using our advanced deep learning models for enhanced detection</p>
                    )}
                </motion.div>

                <div className="analyze-content">
                    {/* Upload Section */}
                    <motion.div
                        className="upload-section"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div
                            {...getRootProps()}
                            className={`dropzone ${isDragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
                        >
                            <input {...getInputProps()} />

                            {preview ? (
                                <div className="preview-container">
                                    <img src={preview} alt="Preview" className="preview-image" />
                                </div>
                            ) : (
                                <div className="dropzone-content">
                                    <div className="dropzone-icon">
                                        {getMediaIcon()}
                                    </div>
                                    {isDragActive ? (
                                        <p>Drop the file here...</p>
                                    ) : file ? (
                                        <div className="file-info">
                                            <p className="file-name">{file.name}</p>
                                            <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                        </div>
                                    ) : (
                                        <>
                                            <p><strong>Drag & drop</strong> your file here</p>
                                            <p className="text-muted">or click to browse</p>
                                            <p className="supported-formats">
                                                Supports: JPG, PNG, MP4, AVI, WAV, MP3
                                            </p>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="actions">
                            {file && !loading && (
                                <>
                                    <button onClick={handleAnalyze} className="btn btn-primary">
                                        Analyze
                                    </button>
                                    <button onClick={handleReset} className="btn btn-secondary">
                                        Clear
                                    </button>
                                </>
                            )}

                            {loading && (
                                <motion.div
                                    className="analysis-loading"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                >
                                    <div className="scanner-container">
                                        {/* Outer ring pulse */}
                                        <div className="scanner-ring ring-1"></div>
                                        <div className="scanner-ring ring-2"></div>
                                        <div className="scanner-ring ring-3"></div>

                                        {/* Center core */}
                                        <div className="scanner-core">
                                            <div className="core-inner">
                                                <Zap className="core-icon" size={32} />
                                            </div>
                                        </div>

                                        {/* Scanning line */}
                                        <div className="scan-line"></div>

                                        {/* Orbiting particles */}
                                        <div className="particle p1"></div>
                                        <div className="particle p2"></div>
                                        <div className="particle p3"></div>
                                        <div className="particle p4"></div>
                                    </div>

                                    <div className="loading-text">
                                        <motion.span
                                            animate={{ opacity: [1, 0.5, 1] }}
                                            transition={{ duration: 1.5, repeat: Infinity }}
                                        >
                                            {mode === 'advanced' ? 'Enhanced Detection Running' : 'Analyzing Media'}
                                        </motion.span>
                                    </div>

                                    <div className="loading-steps">
                                        <motion.div
                                            className="step"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: 0.2 }}
                                        >
                                            ✓ Extracting features...
                                        </motion.div>
                                        <motion.div
                                            className="step active"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: [0.5, 1, 0.5] }}
                                            transition={{ delay: 0.5, duration: 1, repeat: Infinity }}
                                        >
                                            ⟳ Running deep analysis...
                                        </motion.div>
                                        <motion.div
                                            className="step pending"
                                            initial={{ opacity: 0.3 }}
                                            animate={{ opacity: 0.3 }}
                                        >
                                            ○ Generating report...
                                        </motion.div>
                                    </div>
                                </motion.div>
                            )}
                        </div>

                        {error && (
                            <motion.div
                                className="error-message"
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <AlertTriangle size={20} />
                                <span>{error}</span>
                            </motion.div>
                        )}
                    </motion.div>

                    {/* Results Section */}
                    <AnimatePresence>
                        {result && (
                            <motion.div
                                className="results-section"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                            >
                                <div className="results-header">
                                    <h2>Analysis Results</h2>
                                    {getClassificationBadge(result.classification)}
                                </div>

                                {/* Risk Score */}
                                <div className="risk-score-card card">
                                    <div className="risk-score-display">
                                        <div
                                            className="risk-score-circle"
                                            style={{
                                                background: `conic-gradient(${getRiskColor(result.risk_score)} ${result.risk_score * 3.6}deg, var(--bg-tertiary) 0deg)`,
                                            }}
                                        >
                                            <div className="risk-score-inner">
                                                <span className="risk-score-value" style={{ color: getRiskColor(result.risk_score) }}>
                                                    {Math.round(result.risk_score)}
                                                </span>
                                                <span className="risk-score-label">Risk Score</span>
                                            </div>
                                        </div>

                                        <div className="risk-meta">
                                            <div className="meta-item">
                                                <span className="meta-label">Confidence</span>
                                                <span className="meta-value">{result.confidence}</span>
                                            </div>
                                            <div className="meta-item">
                                                <span className="meta-label">Processing Time</span>
                                                <span className="meta-value">{result.processing_time_ms}ms</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Explanation */}
                                {'explanation' in result && result.explanation && (
                                    <div className="explanation-card card">
                                        <h3>Analysis Summary</h3>
                                        <p className="explanation-summary">{result.explanation.summary}</p>

                                        <div className="explanation-factors">
                                            <h4>Key Findings</h4>
                                            <ul>
                                                {result.explanation.factors.map((factor, index) => (
                                                    <li key={index}>{factor}</li>
                                                ))}
                                            </ul>
                                        </div>

                                        <div className="recommendation">
                                            <h4>Recommendation</h4>
                                            <p>{result.explanation.recommendation}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Signals */}
                                {'signals' in result && result.signals && Object.keys(result.signals).length > 0 && (
                                    <div className="signals-card card">
                                        <h3>Detection Signals</h3>
                                        <div className="signals-grid">
                                            {Object.entries(result.signals).map(([name, data]: [string, any]) => (
                                                <div key={name} className="signal-item">
                                                    <div className="signal-header">
                                                        <span className="signal-name">{name}</span>
                                                        <span
                                                            className="signal-score"
                                                            style={{ color: getRiskColor(data.score * 100) }}
                                                        >
                                                            {Math.round(data.score * 100)}%
                                                        </span>
                                                    </div>
                                                    <div className="signal-bar">
                                                        <div
                                                            className="signal-fill"
                                                            style={{
                                                                width: `${data.score * 100}%`,
                                                                background: getRiskColor(data.score * 100)
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* AI Interpretation (for Enhanced mode without explanation) */}
                                {!('explanation' in result && result.explanation) && getInterpretation() && (
                                    <div className="interpretation-card card">
                                        <h3>{getInterpretation()?.title}</h3>
                                        <p className="interpretation-description">{getInterpretation()?.description}</p>

                                        <div className="interpretation-findings">
                                            <h4>Key Findings</h4>
                                            <ul>
                                                {getInterpretation()?.findings.map((finding: string, idx: number) => (
                                                    <li key={idx}>{finding}</li>
                                                ))}
                                            </ul>
                                        </div>

                                        <div className="interpretation-recommendation">
                                            <h4>Recommendation</h4>
                                            <p>{getInterpretation()?.recommendation}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Download Report Button */}
                                <button
                                    className="download-report-btn btn"
                                    onClick={handleDownloadReport}
                                    disabled={downloading}
                                >
                                    {downloading ? (
                                        <>
                                            <Loader className="animate-spin" size={18} />
                                            Generating Report...
                                        </>
                                    ) : (
                                        <>
                                            <Download size={18} />
                                            Download PDF Report
                                        </>
                                    )}
                                </button>

                                {/* Details Toggle */}
                                <button
                                    className="details-toggle"
                                    onClick={() => setShowDetails(!showDetails)}
                                >
                                    {showDetails ? 'Hide' : 'Show'} Technical Details
                                    {showDetails ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                                </button>

                                <AnimatePresence>
                                    {showDetails && (
                                        <motion.div
                                            className="details-card card"
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                        >
                                            <h3>Raw Response</h3>
                                            <pre className="json-display">
                                                {JSON.stringify(result, null, 2)}
                                            </pre>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
