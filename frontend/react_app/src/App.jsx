import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import Register from "./pages/Register";
import Verify from "./pages/Verify";
import Users from "./pages/Users";
import Logs from "./pages/Logs";
import Dashboard from "./pages/Dashboard";
import TimeStarter from "./pages/TimeStarter";
import Inside from "./pages/Inside";
import Login from "./pages/Login";

import "./styles/App.css";

function AuthenticatedLayout({ children }) {
  return (
    <div className="app-layout">
      <Navbar />
      <main className="app-main-content">{children}</main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Routes>
        {/* Public Admin Login Route */}
        <Route path="/login" element={<Login />} />

        {/* Protected Application Routes */}
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <AuthenticatedLayout>
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/register" element={<Register />} />
                  <Route path="/verify" element={<Verify />} />
                  <Route path="/time-starter" element={<TimeStarter />} />
                  <Route path="/inside" element={<Inside />} />
                  <Route path="/users" element={<Users />} />
                  <Route path="/logs" element={<Logs />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </AuthenticatedLayout>
            </ProtectedRoute>
          }
        />
      </Routes>
    </AuthProvider>
  );
}

export default App;