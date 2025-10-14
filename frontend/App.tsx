import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Header from './Header';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './login';
import Module4 from './module4';
import Module4Planning from './module4_planning';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen from-blue-50 to-white bg-gradient-to-b">
          <Header />
          <main className="px-4 py-8">
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={
                <ProtectedRoute>
                  <Module4 />
                </ProtectedRoute>
              } />
              <Route path="/module4" element={
                <ProtectedRoute>
                  <Module4 />
                </ProtectedRoute>
              } />
              <Route path="/module4/planning" element={
                <ProtectedRoute>
                  <Module4Planning />
                </ProtectedRoute>
              } />
              <Route path="/planning" element={
                <ProtectedRoute>
                  <Module4Planning />
                </ProtectedRoute>
              } />
            </Routes>
          </main>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;