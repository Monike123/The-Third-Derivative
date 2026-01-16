import { Link } from 'react-router-dom';
import { Shield, Image, Video, Music, ArrowRight, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import './HomePage.css';

export function HomePage() {
    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero">
                <div className="container">
                    <motion.div
                        className="hero-content"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="hero-badge">
                            <Shield size={16} />
                            <span>AI-Powered Detection</span>
                        </div>

                        <h1>
                            Detect Deepfakes with
                            <span className="text-gradient"> AI Precision</span>
                        </h1>

                        <p className="hero-description">
                            Advanced multi-modal detection system that analyzes images, videos,
                            and audio for signs of manipulation. Protect yourself from synthetic media.
                        </p>

                        <div className="hero-cta">
                            <Link to="/analyze" className="btn btn-primary btn-lg">
                                Start Analyzing
                                <ArrowRight size={20} />
                            </Link>
                        </div>
                    </motion.div>
                </div>

                <div className="hero-glow"></div>
            </section>

            {/* Features Section */}
            <section className="features">
                <div className="container">
                    <motion.h2
                        className="section-title"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                    >
                        Multi-Modal Detection
                    </motion.h2>

                    <div className="features-grid">
                        <motion.div
                            className="feature-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 }}
                        >
                            <div className="feature-icon">
                                <Image size={28} />
                            </div>
                            <h3>Image Analysis</h3>
                            <p>
                                Detect face swaps, GAN-generated faces, and manipulation artifacts
                                using visual and forensic signal analysis.
                            </p>
                        </motion.div>

                        <motion.div
                            className="feature-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.2 }}
                        >
                            <div className="feature-icon">
                                <Video size={28} />
                            </div>
                            <h3>Video Analysis</h3>
                            <p>
                                Frame-by-frame analysis to detect temporal inconsistencies,
                                lip-sync issues, and deepfake artifacts in videos.
                            </p>
                        </motion.div>

                        <motion.div
                            className="feature-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.3 }}
                        >
                            <div className="feature-icon">
                                <Music size={28} />
                            </div>
                            <h3>Audio Analysis</h3>
                            <p>
                                Identify synthetic voices and AI-generated speech using
                                deep audio forensics and spectral analysis.
                            </p>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* How it Works */}
            <section className="how-it-works">
                <div className="container">
                    <h2 className="section-title">How It Works</h2>

                    <div className="steps">
                        <div className="step">
                            <div className="step-number">1</div>
                            <h4>Upload Media</h4>
                            <p>Drop your image, video, or audio file</p>
                        </div>

                        <div className="step-connector"></div>

                        <div className="step">
                            <div className="step-number">2</div>
                            <h4>AI Analysis</h4>
                            <p>Multiple detection models analyze your content</p>
                        </div>

                        <div className="step-connector"></div>

                        <div className="step">
                            <div className="step-number">3</div>
                            <h4>Get Results</h4>
                            <p>View detailed report with risk score</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Detection Signals */}
            <section className="signals">
                <div className="container">
                    <h2 className="section-title">Detection Signals</h2>

                    <div className="signals-grid">
                        {[
                            'Visual pattern analysis',
                            'Frequency domain forensics',
                            'Noise residual analysis',
                            'Face manipulation detection',
                            'Temporal consistency check',
                            'Audio-visual sync analysis'
                        ].map((signal, index) => (
                            <motion.div
                                key={signal}
                                className="signal-item"
                                initial={{ opacity: 0, x: -20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <CheckCircle size={20} className="signal-icon" />
                                <span>{signal}</span>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="container">
                    <div className="cta-card glass-card">
                        <h2>Ready to Detect Deepfakes?</h2>
                        <p>Upload your media and get results in seconds</p>
                        <Link to="/analyze" className="btn btn-primary btn-lg">
                            Analyze Now
                            <ArrowRight size={20} />
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}
