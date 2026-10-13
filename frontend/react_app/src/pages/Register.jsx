import { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import api from "../services/api";
import "../styles/Register.css";

const DEFAULT_TEMPLATES = [
  { id: "face", label: "👤 Identity Photo (Look Straight at Camera)", required: true },
  { id: 1, label: "👁️👁️ Both Eyes - Without Glasses", required: true },
  { id: 2, label: "👓👁️👁️ Both Eyes - With Glasses", required: true },
  { id: 3, label: "👁️ Left Eye - Without Glasses", required: true },
  { id: 4, label: "👓👁️ Left Eye - With Glasses", required: true },
  { id: 5, label: "👁️ Right Eye - Without Glasses", required: true },
  { id: 6, label: "👓👁️ Right Eye - With Glasses", required: true },
];

function Register() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [templates, setTemplates] = useState(DEFAULT_TEMPLATES);
  const [templateImages, setTemplateImages] = useState({}); // { templateId: { dataUrl/blob, file } }

  const [activeCameraId, setActiveCameraId] = useState(null); // ID of template slot currently capturing
  const webcamRef = useRef(null);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const refreshCameraDevices = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === "videoinput");
      setCameraDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        const usbDevice = videoInputs.find(
          (d) => d.label && (d.label.toLowerCase().includes("usb") || !d.label.toLowerCase().includes("integrated"))
        );
        setSelectedDeviceId(usbDevice ? usbDevice.deviceId : videoInputs[0].deviceId);
      }
    } catch (err) {
      console.warn("Could not enumerate camera devices:", err);
    }
  };

  useEffect(() => {
    refreshCameraDevices();
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", refreshCameraDevices);
      return () => {
        navigator.mediaDevices.removeEventListener("devicechange", refreshCameraDevices);
      };
    }
  }, []);

  // ----------------------------------------------------
  // Camera Capture Handler
  // ----------------------------------------------------
  const handleCaptureImage = (templateId) => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      alert("Unable to capture image from camera.");
      return;
    }

    setTemplateImages((prev) => ({
      ...prev,
      [templateId]: { preview: imageSrc, dataUrl: imageSrc },
    }));
    setActiveCameraId(null);
  };

  // ----------------------------------------------------
  // File Upload Handler
  // ----------------------------------------------------
  const handleFileUpload = (templateId, file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setTemplateImages((prev) => ({
        ...prev,
        [templateId]: { preview: reader.result, file: file },
      }));
    };
    reader.readAsDataURL(file);
  };

  // ----------------------------------------------------
  // Add Custom Extra Template Slot
  // ----------------------------------------------------
  const handleAddCustomTemplate = () => {
    const nextId = templates.length + 1;
    const customName = prompt("Enter Custom Template Description (e.g., 'With Tinted Lens' or 'Dim Light Capture'):");
    if (!customName) return;

    setTemplates([
      ...templates,
      { id: nextId, label: `📷 ${customName}`, required: false },
    ]);
  };

  // ----------------------------------------------------
  // Remove Template Slot Image
  // ----------------------------------------------------
  const handleRemoveImage = (templateId) => {
    setTemplateImages((prev) => {
      const copy = { ...prev };
      delete copy[templateId];
      return copy;
    });
  };

  // ----------------------------------------------------
  // Form Submission & Multi-Vector Payload Processing
  // ----------------------------------------------------
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name) {
      setMessage("❌ Please enter Full Name / User ID.");
      return;
    }

    const capturedCount = Object.keys(templateImages).length;
    if (capturedCount === 0) {
      setMessage("❌ Please capture or upload at least 1 iris template image.");
      return;
    }

    setLoading(true);
    setMessage(`⏳ Processing Biometric Identity & Template Registration...`);

    const formData = new FormData();
    formData.append("name", name);
    if (email) formData.append("email", email);

    // Convert iris template data URLs & files into Blob attachments
    const irisKeys = Object.keys(templateImages).filter((k) => k !== "face");
    if (irisKeys.length > 0) {
      for (const key of irisKeys) {
        const item = templateImages[key];
        if (item.file) {
          formData.append("files", item.file);
        } else if (item.dataUrl) {
          const blob = await fetch(item.dataUrl).then((r) => r.blob());
          formData.append("files", blob, `template_${key}.jpg`);
        }
      }
    } else if (templateImages["face"]) {
      // If only identity photo provided, also use it as initial template
      const item = templateImages["face"];
      if (item.file) {
        formData.append("files", item.file);
      } else if (item.dataUrl) {
        const blob = await fetch(item.dataUrl).then((r) => r.blob());
        formData.append("files", blob, "primary_template.jpg");
      }
    }

    try {
      const response = await api.post("/user/add", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Multi-Template Registration Response:", response.data);

      if (response.data.status === "success") {
        const newUserId = response.data.user_id;

        // Register face biometric sample with the same user_id
        const faceItem = templateImages["face"] || templateImages[1];
        if (faceItem) {
          const faceFormData = new FormData();
          if (faceItem.file) {
            faceFormData.append("file", faceItem.file);
          } else if (faceItem.dataUrl) {
            const faceBlob = await fetch(faceItem.dataUrl).then((r) => r.blob());
            faceFormData.append("file", faceBlob, "identity_photo.jpg");
          }

          try {
            await api.post(`/user/${newUserId}/face/register`, faceFormData, {
              headers: { "Content-Type": "multipart/form-data" }
            });
          } catch (fErr) {
            console.warn("Face registration warning:", fErr);
          }
        }

        setMessage(
          `✅ SUCCESS! User '${name}' registered in MySQL with Biometric Identity & Templates! (User ID: #${newUserId})`
        );
        setName("");
        setEmail("");
        setTemplateImages({});
      } else {
        setMessage(`❌ Registration Failed: ${response.data.message}`);
      }
    } catch (err) {
      console.error("Registration Request Error:", err);
      setMessage("❌ Network or Server Error during registration.");
    }

    setLoading(false);
  };

  return (
    <div className="multi-register-container">
      <div className="multi-register-card">
        <div className="header-section">
          <h2>👤 Biometric Multi-Template User Registration</h2>
          <p className="subtitle">
            Capture 6 standard iris templates (With/Without Glasses, Both/Single Eyes) for 100% verification accuracy.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="user-details-grid">
            <div className="input-field">
              <label>Full Name / User ID *</label>
              <input
                type="text"
                placeholder="Example: Adiveppa / Renki (Admin)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>

            <div className="input-field">
              <label>Email Address (Optional)</label>
              <input
                type="email"
                placeholder="admin@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
            <h3 className="section-title" style={{ margin: 0 }}>📷 Biometric Template Collection (6 Standard Templates)</h3>
            {cameraDevices.length > 0 && (
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "0.9rem", color: "#64748b", fontWeight: 600 }}>Camera:</label>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "6px",
                    border: "1px solid #cbd5e1",
                    background: "#ffffff",
                    fontSize: "0.85rem",
                    fontWeight: 500,
                    outline: "none"
                  }}
                >
                  {cameraDevices.map((d, idx) => (
                    <option key={d.deviceId || idx} value={d.deviceId}>
                      {d.label || `Camera ${idx + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="template-grid">
            {templates.map((tpl) => {
              const imgData = templateImages[tpl.id];
              const isCapturing = activeCameraId === tpl.id;

              return (
                <div
                  key={tpl.id}
                  className={`template-card ${imgData ? "captured" : ""} ${isCapturing ? "active-camera" : ""}`}
                >
                  <div className="card-header">
                    <span className="template-label">{tpl.label}</span>
                    {imgData && <span className="status-badge">✅ Captured</span>}
                  </div>

                  {isCapturing ? (
                    <div className="camera-box">
                      <Webcam
                        key={selectedDeviceId || "default"}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        width={280}
                        height={180}
                        onUserMedia={refreshCameraDevices}
                        videoConstraints={
                          selectedDeviceId
                            ? { deviceId: { exact: selectedDeviceId }, width: 1280, height: 720 }
                            : { facingMode: "user" }
                        }
                      />
                      <div className="camera-actions">
                        <button
                          type="button"
                          className="btn-capture"
                          onClick={() => handleCaptureImage(tpl.id)}
                        >
                          📸 Snap Photo
                        </button>
                        <button
                          type="button"
                          className="btn-cancel"
                          onClick={() => setActiveCameraId(null)}
                        >
                          ✖ Cancel
                        </button>
                      </div>
                    </div>
                  ) : imgData ? (
                    <div className="preview-box">
                      <img src={imgData.preview} alt={tpl.label} className="img-preview" />
                      <button
                        type="button"
                        className="btn-retake"
                        onClick={() => handleRemoveImage(tpl.id)}
                      >
                        🗑️ Retake / Remove
                      </button>
                    </div>
                  ) : (
                    <div className="action-box">
                      <button
                        type="button"
                        className="btn-open-cam"
                        onClick={() => setActiveCameraId(tpl.id)}
                      >
                        📸 Live Camera Capture
                      </button>

                      <span className="or-divider">or</span>

                      <label className="file-upload-btn">
                        📁 Upload File
                        <input
                          type="file"
                          accept="image/*"
                          style={{ display: "none" }}
                          onChange={(e) => handleFileUpload(tpl.id, e.target.files[0])}
                        />
                      </label>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="add-template-section">
            <button type="button" className="btn-add-custom" onClick={handleAddCustomTemplate}>
              ➕ Add Custom Extra Template Slot
            </button>
          </div>

          <button
            type="submit"
            className="btn-submit-registration"
            disabled={loading}
          >
            {loading ? "Registering All Templates..." : `🔒 Register User & Persist All Templates in MySQL`}
          </button>

          {message && (
            <div className={`message-banner ${message.startsWith("✅") ? "success" : "error"}`}>
              {message}
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default Register;