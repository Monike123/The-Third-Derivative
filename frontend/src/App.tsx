import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';

function App() {
    return (
        <Router>
            <Header />
            <main style={{ flex: 1 }}>
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/analyze" element={<AnalyzePage />} />
                </Routes>
            </main>
        </Router>
    );
}

export default App;
