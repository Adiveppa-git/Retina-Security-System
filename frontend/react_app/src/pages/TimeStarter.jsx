import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Webcam from "react-webcam";
import api from "../services/api";
import "../styles/TimeStarter.css";

function TimeStarter() {
  const navigate = useNavigate();
  const location = useLocation();

  // Active Session State
  const [session, setSession] = useState(null);
  const [elapsedTimeStr, setElapsedTimeStr] = useState("00:00:00");
  const [checkoutResult, setCheckoutResult] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check-Out Modal / Camera State
  const [showCheckoutModal, setShowCheckoutModal] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState([]);
  const [captureStatusText, setCaptureStatusText] = useState("");
  const webcamRef = useRef(null);
  const capturedBlobsRef = useRef([]);

  // 1. Recover / Load Active Session
  useEffect(() => {
    initSession();
  }, []);

  const initSession = async () => {
    setLoading(true);

    // First check passed location state
    if (location.state && location.state.user_name && location.state.check_in_time) {
      const sessData = {
        user_id: location.state.user_id,
        user_name: location.state.user_name,
        check_in_time: location.state.check_in_time,
        check_in_formatted: location.state.check_in_time_formatted || new Date(location.state.check_in_time).toLocaleTimeString(),
        status: "INSIDE"
      };
      setSession(sessData);
      localStorage.setItem("active_iris_session", JSON.stringify(sessData));
      setLoading(false);
      return;
    }

    // Next check localStorage
    const saved = localStorage.getItem("active_iris_session");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSession(parsed);
      } catch (e) {
        console.error("Parse session error:", e);
      }
    }

    // Query MySQL backend for active session
    try {
      const res = await api.get("/iris/active_session");
      if (res.data && res.data.active) {
        const backendSess = {
          user_id: res.data.user_id,
          user_name: res.data.user_name,
          check_in_time: res.data.check_in_time,
          check_in_formatted: res.data.check_in_formatted,
          status: "INSIDE"
        };
        setSession(backendSess);
        localStorage.setItem("active_iris_session", JSON.stringify(backendSess));
      }
    } catch (err) {
      console.error("Fetch active session error:", err);
    } finally {
      setLoading(false);
    }
  };

  // 2. Live Continuous Elapsed Time Timer
  useEffect(() => {
    if (!session || !session.check_in_time || checkoutResult) return;

    const updateTimer = () => {
      const checkInMs = new Date(session.check_in_time).getTime();
      const nowMs = Date.now();
      const diffSec = Math.max(0, Math.floor((nowMs - checkInMs) / 1000));

      const hrs = Math.floor(diffSec / 3600);
      const mins = Math.floor((diffSec % 3600) / 60);
      const secs = diffSec % 60;

      const pad = (num) => String(num).padStart(2, "0");
      setElapsedTimeStr(`${pad(hrs)}:${pad(mins)}:${pad(secs)}`);
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [session, checkoutResult]);

  // 3. Manual 1-Frame-Per-Click Capture for Check-Out Verification
  const handleCheckoutFrameCapture = async () => {
    const currentCount = capturedBlobsRef.current.length;

    if (currentCount >= 5) {
      return;
    }

    const imageSrc = webcamRef.current ? webcamRef.current.getScreenshot() : null;
    if (!imageSrc) {
      alert("Camera frame capture failed.");
      return;
    }

    try {
      const blob = await fetch(imageSrc).then((res) => res.blob());
      capturedBlobsRef.current.push(blob);

      const frameNumber = capturedBlobsRef.current.length;
      setCapturedFrames((prev) => [
        ...prev,
        { frameNum: frameNumber, src: imageSrc, faceDetected: true }
      ]);

      if (frameNumber < 5) {
        setCaptureStatusText(`Frame ${frameNumber}/5 Captured — Click for Frame ${frameNumber + 1}`);
      } else {
        setCaptureStatusText("5/5 Frames Captured — Executing CHECK-OUT Verification...");
        await submitCheckoutVerification();
      }
    } catch (err) {
      console.error("Frame capture error:", err);
    }
  };

  const submitCheckoutVerification = async () => {
    if (capturedBlobsRef.current.length !== 5) {
      alert("Exactly 5 frames required for verification.");
      return;
    }

    setCapturing(true);
    try {
      const formData = new FormData();
      capturedBlobsRef.current.forEach((b, idx) => {
        formData.append("files", b, `checkout_frame_${idx}.jpg`);
      });
      formData.append("claimed_user_id", session.user_id);
      formData.append("action", "CHECK-OUT");

      const response = await api.post("/iris/verify_sequence_1to1", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      console.log("Check-Out Response:", response.data);

      if (response.data.final_diagnostic_status === "CLAIMED_IDENTITY_VERIFIED") {
        setCheckoutResult(response.data);
        setShowCheckoutModal(false);
        localStorage.removeItem("active_iris_session");
      } else {
        alert(`Check-Out Verification Failed: ${response.data.decision_reason}. Please try again.`);
        resetCheckoutCapture();
      }
    } catch (err) {
      console.error("Check-Out request error:", err);
      alert("Check-out request failed. Ensure backend server is running.");
      resetCheckoutCapture();
    } finally {
      setCapturing(false);
    }
  };

  const resetCheckoutCapture = () => {
    capturedBlobsRef.current = [];
    setCapturedFrames([]);
    setCaptureStatusText("Ready for Frame 1 Capture");
  };

  const openCheckoutModal = () => {
    resetCheckoutCapture();
    setShowCheckoutModal(true);
  };

  if (loading) {
    return (
      <div className="time-starter-container">
        <div className="session-card">
          <h2>Loading Session Status...</h2>
        </div>
      </div>
    );
  }

  // View B: CHECK-OUT COMPLETED STATE
  if (checkoutResult) {
    return (
      <div className="time-starter-container">
        <div className="session-card checkout-complete-card">
          <div className="session-status-header checkout">
            <h2>🔵 CHECK-OUT COMPLETE</h2>
          </div>

          <div className="session-info-body">
            <div className="info-row">
              <span className="info-label">User</span>
              <span className="info-value highlight">{session.user_name} (#{session.user_id})</span>
            </div>

            <div className="info-row">
              <span className="info-label">Check-In Time</span>
              <span className="info-value">{checkoutResult.check_in_time_formatted || session.check_in_formatted || "N/A"}</span>
            </div>

            <div className="info-row">
              <span className="info-label">Check-Out Time</span>
              <span className="info-value text-checkout">{checkoutResult.check_out_time_formatted || new Date().toLocaleTimeString()}</span>
            </div>

            <div className="info-row total-time-row">
              <span className="info-label">Total Time Inside</span>
              <span className="info-value time-display-badge">{checkoutResult.total_duration || elapsedTimeStr}</span>
            </div>

            <div className="info-row">
              <span className="info-label">Status</span>
              <span className="status-pill-checkout">CHECKED OUT</span>
            </div>
          </div>

          <div className="session-actions">
            <button className="btn-session-nav" onClick={() => navigate("/dashboard")}>
              📊 View Security Dashboard
            </button>
            <button className="btn-session-nav secondary" onClick={() => navigate("/verify")}>
              👁️ Back to Verify Iris
            </button>
          </div>
        </div>
      </div>
    );
  }

  // View A: ACTIVE INSIDE SESSION STATE
  if (session) {
    const frameCount = capturedBlobsRef.current.length;

    return (
      <div className="time-starter-container">
        <div className="session-card">
          <div className="session-status-header checkin">
            <h2>🟢 ACCESS GRANTED</h2>
          </div>

          <div className="session-info-body">
            <div className="info-row">
              <span className="info-label">User</span>
              <span className="info-value highlight">{session.user_name} (#{session.user_id})</span>
            </div>

            <div className="info-row">
              <span className="info-label">Check-In Time</span>
              <span className="info-value">{session.check_in_formatted || new Date(session.check_in_time).toLocaleTimeString()}</span>
            </div>

            <div className="info-row">
              <span className="info-label">Current Status</span>
              <span className="status-pill-inside">🟢 CURRENTLY INSIDE</span>
            </div>

            <div className="timer-box">
              <div className="timer-label">⏱️ Time Since Check-In</div>
              <div className="timer-value">{elapsedTimeStr}</div>
            </div>
          </div>

          <div className="session-actions">
            <button className="btn-checkout-trigger" onClick={openCheckoutModal}>
              🔴 CHECK OUT
            </button>
          </div>
        </div>

        {/* CHECK-OUT BIOMETRIC VERIFICATION MODAL */}
        {showCheckoutModal && (
          <div className="modal-overlay">
            <div className="modal-content">
              <h3>👁️ Check-Out Iris Verification</h3>
              <p>User: <strong>{session.user_name} (#{session.user_id})</strong></p>

              <div className="modal-camera-frame">
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  width={380}
                  height={280}
                  videoConstraints={{ facingMode: "user" }}
                />
              </div>

              <div className="modal-status-text">{captureStatusText}</div>

              <div className="modal-thumb-grid">
                {capturedFrames.map((f, i) => (
                  <div key={i} className="thumb-item">
                    <img src={f.src} alt={`Frame ${i+1}`} />
                    <span>F{i+1}</span>
                  </div>
                ))}
              </div>

              <div className="modal-btn-group">
                <button
                  className="btn-modal-capture"
                  onClick={handleCheckoutFrameCapture}
                  disabled={capturing || frameCount >= 5}
                >
                  {frameCount >= 5 ? "Processing..." : `📸 Capture Frame ${frameCount + 1}/5`}
                </button>

                <button
                  className="btn-modal-cancel"
                  onClick={() => setShowCheckoutModal(false)}
                  disabled={capturing}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // View C: NO ACTIVE SESSION STATE
  return (
    <div className="time-starter-container">
      <div className="session-card empty-card">
        <h2>ℹ️ No Active Check-In Session</h2>
        <p>You are not currently checked inside premises.</p>
        <button className="btn-session-nav" onClick={() => navigate("/verify")}>
          👁️ Go to Verify Iris & Check-In
        </button>
      </div>
    </div>
  );
}

export default TimeStarter;
