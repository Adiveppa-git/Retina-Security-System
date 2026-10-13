import { Link } from "react-router-dom";
import { FaUserPlus, FaEye, FaShieldAlt, FaBolt, FaLock } from "react-icons/fa";
import "../styles/Home.css";

function Home() {
  return (
    <div className="home-container">
      <div className="hero-card">
        <div className="hero-badge">
          <FaShieldAlt className="hero-badge-icon" /> AI-POWERED BIOMETRIC ACCESS
        </div>

        <h1 className="hero-title">IRIS SECURITY SYSTEM</h1>

        <h2 className="hero-subtitle">
          AI Based Smart Unauthorized Person Access Detection
        </h2>

        <p className="hero-description">
          Secure authentication using <strong>Anti-Spoofing, Iris Recognition and FastAPI.</strong>
        </p>

        <div className="hero-actions">
          <Link to="/register" className="hero-btn btn-primary">
            <FaUserPlus /> Register User
          </Link>

          <Link
            to="/verify?action=checkin"
            state={{ action: "CHECK-IN", defaultAction: "CHECK-IN" }}
            className="hero-btn btn-checkin-hero"
          >
            🟢 Check-In
          </Link>

          <Link
            to="/verify?action=checkout"
            state={{ action: "CHECK-OUT", defaultAction: "CHECK-OUT" }}
            className="hero-btn btn-checkout-hero"
          >
            🔴 Check-Out
          </Link>

          <Link to="/inside" className="hero-btn btn-secondary">
            🏢 Inside Premises
          </Link>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon-wrapper">
            <FaEye className="feature-icon" />
          </div>
          <h3>Iris Recognition</h3>
          <p>AI-based iris feature extraction and matching.</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon-wrapper">
            <FaShieldAlt className="feature-icon" />
          </div>
          <h3>Anti-Spoofing</h3>
          <p>Detect suspicious/spoofed presentation attempts.</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon-wrapper">
            <FaBolt className="feature-icon" />
          </div>
          <h3>FastAPI Backend</h3>
          <p>Fast communication between frontend and backend.</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon-wrapper">
            <FaLock className="feature-icon" />
          </div>
          <h3>Secure Verification</h3>
          <p>Biometric verification with stored templates.</p>
        </div>
      </div>
    </div>
  );
}

export default Home;