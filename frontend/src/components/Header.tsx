import { Link, useLocation } from 'react-router-dom';
import { Shield, Menu, X } from 'lucide-react';
import { useState } from 'react';
import './Header.css';

export function Header() {
    const location = useLocation();
    const [menuOpen, setMenuOpen] = useState(false);

    return (
        <header className="header">
            <div className="header-content container">
                <Link to="/" className="logo">
                    <div className="logo-icon">
                        <Shield size={28} />
                    </div>
                    <span className="logo-text">
                        <span className="text-gradient">Deepfake</span> Detector
                    </span>
                </Link>

                <nav className={`nav ${menuOpen ? 'open' : ''}`}>
                    <Link
                        to="/"
                        className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
                        onClick={() => setMenuOpen(false)}
                    >
                        Home
                    </Link>
                    <Link
                        to="/analyze"
                        className={`nav-link ${location.pathname === '/analyze' ? 'active' : ''}`}
                        onClick={() => setMenuOpen(false)}
                    >
                        Analyze Media
                    </Link>
                </nav>

                <button className="menu-toggle" onClick={() => setMenuOpen(!menuOpen)}>
                    {menuOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>
        </header>
    );
}
