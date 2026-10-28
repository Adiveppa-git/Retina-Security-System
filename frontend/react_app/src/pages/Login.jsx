import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FaShieldAlt,
  FaUserShield,
  FaLock,
  FaEye,
  FaEyeSlash,
  FaExclamationCircle,
  FaSpinner,
} from "react-icons/fa";
import { useAuth } from "../context/AuthContext";
import "../styles/Login.css";

export default function Login() {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [videoFailed, setVideoFailed] = useState(false);

  // If already authenticated, redirect to destination or home
  useEffect(() => {
    if (isAuthenticated) {
      const dest = location.state?.from?.pathname || "/";
      navigate(dest, { replace: true });
    }
  }, [isAuthenticated, navigate, location]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const trimmedUser = username.trim();
    const trimmedPass = password.trim();

    if (!trimmedUser || !trimmedPass) {
      setError("Please enter both administrator username and password.");
      return;
    }

    setIsSubmitting(true);
    try {
      await login(trimmedUser, trimmedPass);
      const dest = location.state?.from?.pathname || "/";
      navigate(dest, { replace: true });
    } catch (err) {
      const serverMessage =
        err.response?.data?.detail ||
        err.message ||
        "Authentication failed. Please verify your administrator credentials.";
      setError(serverMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="login-viewport">
      {/* Background Animated Video */}
      {!videoFailed && (
        <div className="login-video-wrapper">
          <video
            className="login-video-bg"
            autoPlay
            muted
            loop
            playsInline
            onError={() => setVideoFailed(true)}
          >
            <source src="/irisguard-bg.mp4" type="video/mp4" />
          </video>
        </div>
      )}

      {/* Dark Ambient Overlay */}
      <div className="login-video-overlay" />
      <div className="login-ambient-glow" />

      {/* Login Card Content Container */}
      <div className="login-content-container">
        <div className="login-card">
          <div className="login-header">
            <div className="login-badge">
              <FaShieldAlt style={{ fontSize: "11px" }} /> Secure Access Gateway
            </div>
            <div>
              <div className="login-brand-icon-wrapper">
                <FaShieldAlt className="login-brand-icon" />
              </div>
            </div>
            <h1 className="login-title">NEXGUARD</h1>
            <p className="login-subtitle">Intelligent Iris Security and Access Control</p>
          </div>

          <form className="login-form" onSubmit={handleSubmit} noValidate>
            {error && (
              <div className="login-error-banner" role="alert">
                <FaExclamationCircle className="login-error-icon" />
                <span>{error}</span>
              </div>
            )}

            <div className="login-field-group">
              <label className="login-label" htmlFor="admin-username">
                <FaUserShield /> Admin Username
              </label>
              <div className="login-input-wrapper">
                <FaUserShield className="login-input-icon" />
                <input
                  id="admin-username"
                  type="text"
                  className="login-input"
                  placeholder="Enter admin username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  autoFocus
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div className="login-field-group">
              <label className="login-label" htmlFor="admin-password">
                <FaLock /> Password
              </label>
              <div className="login-input-wrapper">
                <FaLock className="login-input-icon" />
                <input
                  id="admin-password"
                  type={showPassword ? "text" : "password"}
                  className="login-input"
                  placeholder="Enter admin password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  className="login-toggle-password-btn"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  title={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="login-submit-btn"
              disabled={isSubmitting}
              id="admin-login-submit"
            >
              {isSubmitting ? (
                <>
                  <FaSpinner className="login-spinner" /> Verifying Credentials...
                </>
              ) : (
                <>Sign In to Console</>
              )}
            </button>
          </form>

          <div className="login-footer">
            Protected by <span className="login-security-tag">NEXGUARD</span> Multi-Factor Biometric Defense
            <br />
            Authorized Security Personnel Only
          </div>
        </div>
      </div>
    </div>
  );
}
