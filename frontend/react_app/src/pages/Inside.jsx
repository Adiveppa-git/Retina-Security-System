import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Webcam from "react-webcam";
import api from "../services/api";
import "../styles/Inside.css";

function Inside() {
  const navigate = useNavigate();
  const location = useLocation();

  const [activeUsers, setActiveUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [now, setNow] = useState(Date.now());

  // Check-Out Modal / Camera State
  const [checkoutUser, setCheckoutUser] = useState(null);
  const [showCheckoutModal, setShowCheckoutModal] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState([]);
  const [captureStatusText, setCaptureStatusText] = useState("");
  const [checkoutError, setCheckoutError] = useState(null);

  // Completed Session Banner State
  const [completedSession, setCompletedSession] = useState(null);
  const [newCheckinUser, setNewCheckinUser] = useState(null);
  const [completedVisits, setCompletedVisits] = useState([]);

  const webcamRef = useRef(null);
  const capturedBlobsRef = useRef([]);

  useEffect(() => {
    if (location.state?.checkoutCompleted && location.state?.checkoutData) {
      const data = location.state.checkoutData;
      setCompletedSession({
        user_name: data.claimed_user_name,
        user_id: data.claimed_user_id,
        check_in_formatted: data.check_in_time_formatted || "N/A",
        check_out_formatted: data.check_out_time_formatted || new Date().toLocaleTimeString(),
        total_duration: data.total_duration || "00h 00m 00s"
      });
      setNewCheckinUser(null);
    } else if (location.state?.checkinCompleted && location.state?.user_name) {
      setNewCheckinUser({
        user_name: location.state.user_name,
        user_id: location.state.user_id,
        check_in_formatted: location.state.check_in_formatted || new Date().toLocaleTimeString()
      });
      setCompletedSession(null);
    }
    fetchActiveSessions();
    fetchCompletedVisits();

    const fetchInterval = setInterval(() => {
      fetchActiveSessions();
      fetchCompletedVisits();
    }, 5000);
    const clockInterval = setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => {
      clearInterval(fetchInterval);
      clearInterval(clockInterval);
    };
  }, [location.state]);

  const fetchCompletedVisits = async () => {
    try {
      const response = await api.get("/iris/completed_visits");
      if (response.data && Array.isArray(response.data.completed_visits)) {
        setCompletedVisits(response.data.completed_visits);
      }
    } catch (err) {
      console.error("Fetch completed visits error:", err);
    }
  };

  const fetchActiveSessions = async () => {
    try {
      // Clean up any stale diagnostic localStorage key
      try { localStorage.removeItem("active_iris_session"); } catch (e) {}

      const response = await api.get("/iris/active_sessions");
      let users = (response.data && Array.isArray(response.data.active_users)) ? response.data.active_users : [];

      // If just navigated from a successful checkout, guarantee checked-out user is not in active list
      if (location.state?.checkoutCompleted && location.state?.checkoutData?.claimed_user_id) {
        users = users.filter((u) => u.user_id !== location.state.checkoutData.claimed_user_id);
      }

      // If just navigated from a successful check-in, ensure user appears immediately
      if (location.state?.checkinCompleted && location.state?.user_name) {
        const found = users.some(u => String(u.user_name).toLowerCase() === String(location.state.user_name).toLowerCase());
        if (!found) {
          users = [
            {
              user_id: location.state.user_id,
              user_name: location.state.user_name,
              check_in_time: location.state.check_in_time || new Date().toISOString(),
              check_in_formatted: location.state.check_in_formatted || new Date().toLocaleTimeString(),
              status: "INSIDE"
            },
            ...users
          ];
        }
      }

      setActiveUsers(users);
    } catch (error) {
      console.error("Fetch active sessions error:", error);
      setActiveUsers([]);
    } finally {
      setLoading(false);
    }
  };

  const calculateDuration = (checkInIso) => {
    if (!checkInIso) return "00:00:00";
    const startMs = new Date(checkInIso).getTime();
    const diffSec = Math.max(0, Math.floor((now - startMs) / 1000));

    const hrs = Math.floor(diffSec / 3600);
    const mins = Math.floor((diffSec % 3600) / 60);
    const secs = diffSec % 60;

    const pad = (num) => String(num).padStart(2, "0");
    return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
  };

  const openCheckoutModal = (user) => {
    setCheckoutUser(user);
    setCheckoutError(null);
    capturedBlobsRef.current = [];
    setCapturedFrames([]);
    setCaptureStatusText("Ready to start 5-frame Check-Out verification");
    setShowCheckoutModal(true);
  };

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Guided 5-Frame Check-Out Verification
  const startGuidedCheckoutCapture = async () => {
    if (!webcamRef.current) {
      alert("Camera unavailable.");
      return;
    }

    setCapturing(true);
    setCheckoutError(null);
    capturedBlobsRef.current = [];
    setCapturedFrames([]);

    const poseInstructions = [
      "Frame 1/5 — Look straight 📸",
      "Frame 2/5 — Turn head slightly LEFT 👈",
      "Frame 3/5 — Turn head slightly RIGHT 👉",
      "Frame 4/5 — Look slightly UP 👆",
      "Frame 5/5 — Look slightly DOWN 👇"
    ];

    try {
      for (let i = 0; i < 5; i++) {
        setCaptureStatusText(poseInstructions[i]);
        if (i > 0) await delay(350);

        let imageSrc = webcamRef.current ? webcamRef.current.getScreenshot() : null;
        let retries = 0;
        while (!imageSrc && retries < 3) {
          await delay(200);
          imageSrc = webcamRef.current ? webcamRef.current.getScreenshot() : null;
          retries++;
        }

        if (!imageSrc) {
          setCheckoutError("Camera frame capture failed.");
          setCapturing(false);
          return;
        }

        const blob = await fetch(imageSrc).then((res) => res.blob());
        capturedBlobsRef.current.push(blob);
        setCapturedFrames((prev) => [...prev, { frameNum: i + 1, src: imageSrc }]);
      }

      if (capturedBlobsRef.current.length !== 5) {
        setCheckoutError("Incomplete capture. Require exactly 5 frames.");
        setCapturing(false);
        return;
      }

      setCaptureStatusText("5/5 Frames Captured — Processing Check-Out Verification...");

      const formData = new FormData();
      capturedBlobsRef.current.forEach((b, idx) => {
        formData.append("files", b, `checkout_frame_${idx}.jpg`);
      });
      formData.append("claimed_user_id", checkoutUser.user_id);
      formData.append("action", "CHECK-OUT");

      const response = await api.post("/iris/verify_sequence_1to1", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      console.log("Check-Out Verification Response:", response.data);

      if (response.data.final_diagnostic_status === "CLAIMED_IDENTITY_VERIFIED") {
        setShowCheckoutModal(false);
        setCompletedSession({
          user_name: response.data.claimed_user_name,
          user_id: response.data.claimed_user_id,
          check_in_formatted: response.data.check_in_time_formatted || checkoutUser.check_in_formatted || "N/A",
          check_out_formatted: response.data.check_out_time_formatted || new Date().toLocaleTimeString(),
          total_duration: response.data.total_duration || calculateDuration(checkoutUser.check_in_time)
        });
        localStorage.removeItem("active_iris_session");
        fetchActiveSessions();
        fetchCompletedVisits();
      } else {
        setCheckoutError(`Check-Out Verification Failed: ${response.data.decision_reason || "Biometric mismatch"}. User remains INSIDE.`);
        setCaptureStatusText("Verification Failed — Session Retained");
      }
    } catch (err) {
      console.error("Check-out verification error:", err);
      setCheckoutError("Check-out request failed. Ensure backend server is running.");
    } finally {
      setCapturing(false);
    }
  };

  const handleRedirectToVerifyCheckout = (user) => {
    navigate("/verify?action=checkout", {
      state: {
        action: "CHECK-OUT",
        defaultAction: "CHECK-OUT",
        user_id: user?.user_id,
        user_name: user?.user_name
      }
    });
  };

  if (loading) {
    return (
      <div className="inside-page-container">
        <div className="inside-card">
          <div className="inside-loading-box">
            <h2>Loading Active Inside Sessions...</h2>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="inside-page-container">
      {/* COMPLETED CHECK-OUT BANNER CARD */}
      {completedSession && (
        <div className="completed-checkout-banner">
          <div className="completed-banner-header">
            <h3>🔴 CHECK-OUT COMPLETED SUCCESSFULLY</h3>
            <button className="btn-close-banner" onClick={() => setCompletedSession(null)}>✕</button>
          </div>
          <div className="completed-banner-body">
            <div className="completed-info-row">
              <span className="label">User:</span>
              <span className="value bold">{completedSession.user_name} (#{completedSession.user_id})</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Check-In Time:</span>
              <span className="value">{completedSession.check_in_formatted}</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Check-Out Time:</span>
              <span className="value text-checkout">{completedSession.check_out_formatted}</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Total Duration:</span>
              <span className="value badge-duration">{completedSession.total_duration}</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Status:</span>
              <span className="badge-checkout-tag">🔴 CHECKED OUT</span>
            </div>
          </div>
        </div>
      )}

      {/* NEW CHECK-IN SUCCESS BANNER */}
      {newCheckinUser && (
        <div className="completed-checkout-banner" style={{ borderColor: "#10b981", background: "rgba(16, 185, 129, 0.08)" }}>
          <div className="completed-banner-header">
            <h3 style={{ color: "#10b981" }}>🟢 CHECK-IN COMPLETED — ACCESS GRANTED</h3>
            <button className="btn-close-banner" onClick={() => setNewCheckinUser(null)}>✕</button>
          </div>
          <div className="completed-banner-body">
            <div className="completed-info-row">
              <span className="label">User:</span>
              <span className="value bold">{newCheckinUser.user_name} (#{newCheckinUser.user_id})</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Entry Time:</span>
              <span className="value">{newCheckinUser.check_in_formatted}</span>
            </div>
            <div className="completed-info-row">
              <span className="label">Status:</span>
              <span className="badge-inside-tag">🟢 INSIDE</span>
            </div>
          </div>
        </div>
      )}

      {/* MAIN ACTIVE INSIDE CARD */}
      <div className="inside-card">
        {activeUsers.length === 0 ? (
          <div className="empty-inside-view">
            <div className="inside-header empty">
              <h2>🟢 NO USERS INSIDE</h2>
            </div>
            <div className="empty-inside-body">
              <div className="empty-icon">🏢</div>
              <p>No active check-in sessions.</p>
              <small>Users who complete a successful iris check-in will appear here live with dynamic duration timers.</small>
              <button
                className="btn-go-verify"
                onClick={() => navigate("/verify?action=checkin", { state: { action: "CHECK-IN", defaultAction: "CHECK-IN" } })}
              >
                👁️ Go to Verify & Check-In
              </button>
            </div>
          </div>
        ) : (
          <div className="active-inside-view">
            <div className="inside-header active">
              <h2>🟢 USERS INSIDE PREMISES</h2>
              <span className="active-count-tag">{activeUsers.length} Active</span>
            </div>

            <div className="table-responsive">
              <table className="inside-table">
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Check-In Time</th>
                    <th>Duration</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {activeUsers.map((user) => (
                    <tr key={user.user_id}>
                      <td className="user-name-cell">
                        <strong>{user.user_name}</strong>
                        <span className="user-id-sub">#{user.user_id}</span>
                      </td>
                      <td>{user.check_in_formatted || "N/A"}</td>
                      <td className="duration-cell">
                        {calculateDuration(user.check_in_time)}
                      </td>
                      <td>
                        <span className="badge-inside-tag">🟢 INSIDE</span>
                      </td>
                      <td>
                        <button
                          className="btn-checkout-table"
                          onClick={() => handleRedirectToVerifyCheckout(user)}
                        >
                          🔴 CHECK OUT
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="inside-footer-actions">
              <button
                className="btn-go-verify"
                onClick={() => navigate("/verify?action=checkin", { state: { action: "CHECK-IN", defaultAction: "CHECK-IN" } })}
              >
                🟢 Verify & Check-In New User
              </button>
              <button
                className="btn-go-verify"
                onClick={() => navigate("/verify?action=checkout", { state: { action: "CHECK-OUT", defaultAction: "CHECK-OUT" } })}
                style={{ marginLeft: "12px", background: "linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)" }}
              >
                🔴 Verify & Check-Out
              </button>
            </div>
          </div>
        )}
      </div>

      {/* COMPLETED VISITS HISTORY */}
      {completedVisits && completedVisits.length > 0 && (
        <div className="inside-card" style={{ marginTop: "24px" }}>
          <div className="inside-header" style={{ background: "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)", borderBottom: "1px solid #334155" }}>
            <h2>📋 COMPLETED VISITS & DURATION HISTORY</h2>
            <span className="active-count-tag" style={{ background: "rgba(59, 130, 246, 0.2)", color: "#60a5fa" }}>
              {completedVisits.length} Visits
            </span>
          </div>
          <div className="table-responsive">
            <table className="inside-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Check-In Time</th>
                  <th>Check-Out Time</th>
                  <th>Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {completedVisits.map((v, idx) => (
                  <tr key={idx}>
                    <td className="user-name-cell">
                      <strong>{v.user_name}</strong>
                    </td>
                    <td>{v.check_in_formatted || "N/A"}</td>
                    <td>{v.check_out_formatted || "N/A"}</td>
                    <td className="duration-cell" style={{ color: "#38bdf8", fontWeight: "bold" }}>
                      {v.duration || "00h 00m 00s"}
                    </td>
                    <td>
                      <span className="badge-checkout-tag">🔴 COMPLETED</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* BIOMETRIC CHECK-OUT MODAL */}
      {showCheckoutModal && checkoutUser && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>👁️ Check-Out Biometric Verification</h3>
              <button className="btn-modal-close" onClick={() => setShowCheckoutModal(false)} disabled={capturing}>✕</button>
            </div>

            <div className="modal-user-info">
              User: <strong>{checkoutUser.user_name} (#{checkoutUser.user_id})</strong>
            </div>

            <div className="modal-camera-wrapper">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                width={360}
                height={260}
                videoConstraints={{ facingMode: "user" }}
              />
            </div>

            <div className="modal-status-text">{captureStatusText}</div>

            {checkoutError && (
              <div className="modal-error-alert">
                🔴 {checkoutError}
              </div>
            )}

            <div className="modal-thumb-grid">
              {capturedFrames.map((f, i) => (
                <div key={i} className="thumb-item">
                  <img src={f.src} alt={`Frame ${i + 1}`} />
                  <span>F{i + 1}</span>
                </div>
              ))}
            </div>

            <div className="modal-btn-group">
              <button
                className="btn-modal-capture"
                onClick={startGuidedCheckoutCapture}
                disabled={capturing}
              >
                {capturing ? "Capturing 5 Frames..." : "📸 Start 5-Frame Check-Out Verification"}
              </button>

              <button
                className="btn-modal-redirect"
                onClick={() => handleRedirectToVerifyCheckout(checkoutUser)}
                disabled={capturing}
              >
                👁️ Open Main Verification Page
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

export default Inside;
