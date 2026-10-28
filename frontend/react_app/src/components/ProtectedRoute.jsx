import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { FaShieldAlt } from "react-icons/fa";

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#060b16",
          color: "#94a3b8",
          fontFamily: "'Segoe UI', Roboto, sans-serif",
          gap: "16px",
        }}
      >
        <FaShieldAlt
          style={{
            fontSize: "48px",
            color: "#38bdf8",
            filter: "drop-shadow(0 0 16px rgba(56, 189, 248, 0.6))",
            animation: "pulse 1.8s infinite ease-in-out",
          }}
        />
        <style>{`
          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 1; filter: drop-shadow(0 0 24px rgba(56, 189, 248, 0.9)); }
          }
        `}</style>
        <div style={{ fontSize: "16px", fontWeight: "600", letterSpacing: "1px", color: "#f1f5f9" }}>
          IRISGUARD ACCESS CONTROL
        </div>
        <div style={{ fontSize: "13px", color: "#64748b" }}>Verifying secure admin session...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children;
}
