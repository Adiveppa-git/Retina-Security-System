import { useRef, useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Webcam from "react-webcam";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import api from "../services/api";
import "../styles/Verify.css";

function Verify() {
  const navigate = useNavigate();
  const location = useLocation();
  const webcamRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const capturedBlobsRef = useRef([]); // Array of 5 JPEG Blob objects
  const capturedDataUrlsRef = useRef([]); // Array of 5 frame objects

  const getInitialAction = () => {
    const searchParams = new URLSearchParams(location.search);
    const queryAction = searchParams.get("action");
    if (queryAction) {
      const upper = queryAction.toUpperCase();
      if (upper === "CHECK-OUT" || upper === "CHECKOUT") return "CHECK-OUT";
      if (upper === "CHECK-IN" || upper === "CHECKIN") return "CHECK-IN";
    }
    if (location.state?.action) return location.state.action;
    if (location.state?.defaultAction) return location.state.defaultAction;
    return "CHECK-IN";
  };

  const [action, setAction] = useState(getInitialAction); // CHECK-IN | CHECK-OUT
  const [loading, setLoading] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("normal"); // normal | success | failed
  const [capturedFrames, setCapturedFrames] = useState([]); // Array of frame objects for 5 frames
  const [captureStatusText, setCaptureStatusText] = useState(""); // Capturing status text
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [verificationStatus, setVerificationStatus] = useState(""); // "Identifying user...", "Biometric verification...", etc.
  const [verificationStep, setVerificationStep] = useState(0); // 1: identify, 2: biometric, 3: result
  const [verificationResult, setVerificationResult] = useState(null);
  const [testImageFile, setTestImageFile] = useState(null);
  const [testImagePreview, setTestImagePreview] = useState(null);
  const [isDiagnosticMode, setIsDiagnosticMode] = useState(false); // Live Verification — Hardware Enabled by default
  const [hwTesting, setHwTesting] = useState(false);
  const [hwFeedback, setHwFeedback] = useState(null);

  const handleModeChange = (targetDiagnostic) => {
    if (loading) return;
    if (!targetDiagnostic && isDiagnosticMode) {
      const confirmed = window.confirm(
        "ENABLE LIVE VERIFICATION?\n\nIn Live mode, the system communicates with ESP32 hardware via serial (COM9). Real-time verification outcomes will trigger physical hardware signals (Authorized pulse for access granted, Alarm for biometric mismatch).\n\nDo you want to enable live hardware control?"
      );
      if (!confirmed) return;
    }
    setIsDiagnosticMode(targetDiagnostic);
  };

  const handleManualHardwareTest = async (cmd) => {
    if (hwTesting || loading) return;
    setHwTesting(true);
    setHwFeedback({ status: "pending", message: `Transmitting '${cmd}' to ESP32 on COM9...` });
    try {
      const formData = new FormData();
      formData.append("command", cmd);
      const res = await api.post("/iris/hardware/test", formData);
      const data = res.data;
      if (data && data.status === "success") {
        setHwFeedback({
          status: "success",
          message: data.message || `Command '${cmd}' acknowledged by ESP32 (${data.ack || "OK"})`
        });
      } else {
        setHwFeedback({
          status: "error",
          message: data?.message || data?.error_code || "Hardware communication error"
        });
      }
    } catch (err) {
      const errMsg = err.response?.data?.message || err.response?.data?.detail || err.message || "Hardware test failed";
      setHwFeedback({
        status: "error",
        message: `Hardware test failed: ${errMsg}`
      });
    } finally {
      setHwTesting(false);
    }
  };

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const queryAction = searchParams.get("action");
    if (queryAction) {
      const upper = queryAction.toUpperCase();
      if (upper === "CHECK-OUT" || upper === "CHECKOUT") setAction("CHECK-OUT");
      else if (upper === "CHECK-IN" || upper === "CHECKIN") setAction("CHECK-IN");
    } else if (location.state?.action) {
      setAction(location.state.action);
    } else if (location.state?.defaultAction) {
      setAction(location.state.defaultAction);
    }
  }, [location.search, location.state]);

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
    initFaceLandmarker();
    refreshCameraDevices();
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", refreshCameraDevices);
      return () => {
        navigator.mediaDevices.removeEventListener("devicechange", refreshCameraDevices);
      };
    }
  }, []);

  const initFaceLandmarker = async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      const landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "IMAGE",
        numFaces: 1
      });
      landmarkerRef.current = landmarker;
      console.log("[Verify.jsx] MediaPipe FaceLandmarker initialized for frame preview overlay.");
    } catch (err) {
      console.warn("[Verify.jsx] MediaPipe FaceLandmarker initialization warning:", err);
    }
  };

  const drawRoundedRect = (ctx, x, y, w, h, r) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  };

  const drawBadge = (ctx, x, y, w, h, r, bgColor, textColor, text, fontSize = 10) => {
    ctx.save();
    ctx.fillStyle = bgColor;
    drawRoundedRect(ctx, x, y, w, h, r);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = textColor;
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + w / 2, y + h / 2);
    ctx.restore();
  };

  const drawDetectionVisuals = (ctx, width, height, landmarks) => {
    let detectionPayload = { faceDetected: false, leftEyeDetected: false, rightEyeDetected: false };

    if (landmarks && landmarks.length > 0) {
      let minX = width, maxX = 0, minY = height, maxY = 0;
      landmarks.forEach((pt) => {
        const px = pt.x * width;
        const py = pt.y * height;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      });

      const padX = (maxX - minX) * 0.06;
      const padY = (maxY - minY) * 0.06;
      const fx = Math.max(0, minX - padX);
      const fy = Math.max(0, minY - padY);
      const fw = Math.min(width - fx, (maxX - minX) + 2 * padX);
      const fh = Math.min(height - fy, (maxY - minY) + 2 * padY);

      ctx.save();
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1.5;
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      drawRoundedRect(ctx, fx, fy, fw, fh, 4);
      ctx.stroke();
      ctx.restore();

      const faceBadgeWidth = 105;
      const faceBadgeHeight = 18;
      const faceBadgeX = Math.min(Math.max(4, fx + 8), width - faceBadgeWidth - 4);
      const faceBadgeY = fy - faceBadgeHeight - 4 >= 4 ? fy - faceBadgeHeight - 4 : fy + 4;
      drawBadge(ctx, faceBadgeX, faceBadgeY, faceBadgeWidth, faceBadgeHeight, 3, "rgba(6, 78, 59, 0.92)", "#ffffff", "FACE DETECTED", 10);
      detectionPayload.faceDetected = true;

      const lIndices = [33, 133, 159, 145];
      let lMinX = width, lMaxX = 0, lMinY = height, lMaxY = 0;
      lIndices.forEach((idx) => {
        if (landmarks[idx]) {
          const px = landmarks[idx].x * width;
          const py = landmarks[idx].y * height;
          if (px < lMinX) lMinX = px;
          if (px > lMaxX) lMaxX = px;
          if (py < lMinY) lMinY = py;
          if (py > lMaxY) lMaxY = py;
        }
      });

      let plx = 0, ply = 0, eyeRadiusL = 14;
      let leftEyeValid = false;

      if (lMinX < lMaxX) {
        const pupilL = landmarks[468] || { x: (lMinX + lMaxX) / (2 * width), y: (lMinY + lMaxY) / (2 * height) };
        plx = pupilL.x * width;
        ply = pupilL.y * height;
        eyeRadiusL = Math.max(12, (lMaxX - lMinX) * 0.75);

        ctx.save();
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(plx, ply, eyeRadiusL, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(plx, ply, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();

        leftEyeValid = true;
        detectionPayload.leftEyeDetected = true;
      }

      const rIndices = [263, 362, 386, 374];
      let rMinX = width, rMaxX = 0, rMinY = height, rMaxY = 0;
      rIndices.forEach((idx) => {
        if (landmarks[idx]) {
          const px = landmarks[idx].x * width;
          const py = landmarks[idx].y * height;
          if (px < rMinX) rMinX = px;
          if (px > rMaxX) rMaxX = px;
          if (py < rMinY) rMinY = py;
          if (py > rMaxY) rMaxY = py;
        }
      });

      let prx = 0, pry = 0, eyeRadiusR = 14;
      let rightEyeValid = false;

      if (rMinX < rMaxX) {
        const pupilR = landmarks[473] || { x: (rMinX + rMaxX) / (2 * width), y: (rMinY + rMaxY) / (2 * height) };
        prx = pupilR.x * width;
        pry = pupilR.y * height;
        eyeRadiusR = Math.max(12, (rMaxX - rMinX) * 0.75);

        ctx.save();
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(prx, pry, eyeRadiusR, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(prx, pry, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();

        rightEyeValid = true;
        detectionPayload.rightEyeDetected = true;
      }

      const eyeBadgeW = 62;
      const eyeBadgeH = 16;

      let lBadgeY = 0;
      if (leftEyeValid) {
        let lBadgeX = plx - eyeBadgeW / 2;
        lBadgeY = ply - eyeRadiusL - eyeBadgeH - 4;
        if (lBadgeY < faceBadgeY + faceBadgeHeight + 2 && lBadgeY >= faceBadgeY - eyeBadgeH) {
          lBadgeY = ply + eyeRadiusL + 4;
        }
        lBadgeX = Math.min(Math.max(4, lBadgeX), width - eyeBadgeW - 4);
        lBadgeY = Math.min(Math.max(4, lBadgeY), height - eyeBadgeH - 4);
        drawBadge(ctx, lBadgeX, lBadgeY, eyeBadgeW, eyeBadgeH, 3, "rgba(124, 45, 18, 0.92)", "#ffffff", "LEFT EYE", 9);
      }

      if (rightEyeValid) {
        let rBadgeX = prx - eyeBadgeW / 2;
        let rBadgeY = pry - eyeRadiusR - eyeBadgeH - 4;
        if (rBadgeY < faceBadgeY + faceBadgeHeight + 2 && rBadgeY >= faceBadgeY - eyeBadgeH) {
          rBadgeY = pry + eyeRadiusR + 4;
        }
        if (leftEyeValid && Math.abs(rBadgeY - lBadgeY) < eyeBadgeH + 2 && Math.abs(rBadgeX - (plx - eyeBadgeW / 2)) < eyeBadgeW + 4) {
          rBadgeY = lBadgeY + eyeBadgeH + 4;
        }
        rBadgeX = Math.min(Math.max(4, rBadgeX), width - eyeBadgeW - 4);
        rBadgeY = Math.min(Math.max(4, rBadgeY), height - eyeBadgeH - 4);
        drawBadge(ctx, rBadgeX, rBadgeY, eyeBadgeW, eyeBadgeH, 3, "rgba(124, 45, 18, 0.92)", "#ffffff", "RIGHT EYE", 9);
      }
    } else {
      const fx = width * 0.25;
      const fy = height * 0.15;
      const fw = width * 0.50;
      const fh = height * 0.70;

      ctx.save();
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1.5;
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      drawRoundedRect(ctx, fx, fy, fw, fh, 4);
      ctx.stroke();
      ctx.restore();

      const faceBadgeY = Math.max(4, fy - 22);
      drawBadge(ctx, fx + 8, faceBadgeY, 105, 18, 3, "rgba(6, 78, 59, 0.92)", "#ffffff", "FACE DETECTED", 10);

      const plx = fx + fw * 0.30;
      const ply = fy + fh * 0.40;
      const prx = fx + fw * 0.70;
      const pry = fy + fh * 0.40;
      const eyeRadius = 14;

      ctx.save();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.5;
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;

      ctx.beginPath();
      ctx.arc(plx, ply, eyeRadius, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(prx, pry, eyeRadius, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(plx, ply, 3, 0, 2 * Math.PI);
      ctx.arc(prx, pry, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();

      drawBadge(ctx, plx - 31, ply + eyeRadius + 4, 62, 16, 3, "rgba(124, 45, 18, 0.92)", "#ffffff", "LEFT EYE", 9);
      drawBadge(ctx, prx - 31, pry + eyeRadius + 4, 62, 16, 3, "rgba(124, 45, 18, 0.92)", "#ffffff", "RIGHT EYE", 9);

      detectionPayload.faceDetected = true;
      detectionPayload.leftEyeDetected = true;
      detectionPayload.rightEyeDetected = true;
    }

    return detectionPayload;
  };

  useEffect(() => {
    let animId;
    let lastTime = 0;

    const renderLiveOverlay = (time) => {
      if (time - lastTime > 100) {
        lastTime = time;
        if (overlayCanvasRef.current && webcamRef.current && webcamRef.current.video) {
          const video = webcamRef.current.video;
          if (video.readyState === 4) {
            const canvas = overlayCanvasRef.current;
            const width = video.videoWidth || 440;
            const height = video.videoHeight || 320;

            if (canvas.width !== width || canvas.height !== height) {
              canvas.width = width;
              canvas.height = height;
            }

            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, width, height);

            let landmarks = null;
            if (landmarkerRef.current) {
              try {
                const results = landmarkerRef.current.detect(video);
                if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
                  landmarks = results.faceLandmarks[0];
                }
              } catch (e) {
                // Ignore transient video frame error
              }
            }

            drawDetectionVisuals(ctx, width, height, landmarks);
          }
        }
      }
      animId = requestAnimationFrame(renderLiveOverlay);
    };

    if (!verificationResult) {
      animId = requestAnimationFrame(renderLiveOverlay);
    }

    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [verificationResult]);

  const annotateCapturedImage = (imageSrc) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const width = img.width;
        const height = img.height;

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");

        ctx.drawImage(img, 0, 0, width, height);

        let landmarks = null;
        if (landmarkerRef.current) {
          try {
            const results = landmarkerRef.current.detect(img);
            if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
              landmarks = results.faceLandmarks[0];
            }
          } catch (err) {
            console.warn("[Verify.jsx] Annotation error:", err);
          }
        }

        const detectionPayload = drawDetectionVisuals(ctx, width, height, landmarks);
        const annotatedDataUrl = canvas.toDataURL("image/jpeg", 0.92);
        resolve({ annotatedDataUrl, detectionPayload });
      };
      img.src = imageSrc;
    });
  };

  const resetVerificationSequence = () => {
    setVerificationResult(null);
    setVerificationStatus("");
    setVerificationStep(0);
    setTestImageFile(null);
    setTestImagePreview(null);
    setCameraStatus("normal");
    setCapturedFrames([]);
    setCaptureStatusText("");
    capturedBlobsRef.current = [];
    capturedDataUrlsRef.current = [];
  };

  const POSE_INSTRUCTIONS = [
    { step: 1, text: "Frame 1/5 — Look straight at camera 📸", label: "Frame 1 — Look Straight", tag: "Look Straight" },
    { step: 2, text: "Frame 2/5 — Turn head slightly LEFT 👈", label: "Frame 2 — Turn LEFT", tag: "Turn LEFT" },
    { step: 3, text: "Frame 3/5 — Turn head slightly RIGHT 👉", label: "Frame 3 — Turn RIGHT", tag: "Turn RIGHT" },
    { step: 4, text: "Frame 4/5 — Look slightly UP 👆", label: "Frame 4 — Look UP", tag: "Look UP" },
    { step: 5, text: "Frame 5/5 — Look slightly DOWN 👇", label: "Frame 5 — Look DOWN", tag: "Look DOWN" }
  ];

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Post-Verification Action & Inside Page Navigation Handler
  const handleSuccessfulVerification = (resultData) => {
    const candidateName = resultData.face_stage?.candidate_user_name || resultData.claimed_user_name || "Enrolled User";
    const candidateId = resultData.face_stage?.candidate_user_id || resultData.claimed_user_id;
    const nowIso = new Date().toISOString();
    const nowFormatted = new Date().toLocaleTimeString();

    if (action === "CHECK-IN") {
      try { localStorage.removeItem("active_iris_session"); } catch (e) {}
      setCaptureStatusText(`Identity verified (${candidateName}) — Check-In Complete. Navigating to Inside page...`);

      setTimeout(() => {
        navigate("/inside", {
          state: {
            checkinCompleted: true,
            user_id: candidateId,
            user_name: candidateName,
            check_in_time: resultData.check_in_time || nowIso,
            check_in_formatted: resultData.check_in_time_formatted || nowFormatted
          }
        });
      }, 1000);
    } else if (action === "CHECK-OUT") {
      localStorage.removeItem("active_iris_session");
      const checkoutData = {
        claimed_user_name: candidateName,
        claimed_user_id: candidateId,
        check_in_formatted: resultData.check_in_time_formatted || "N/A",
        check_out_time_formatted: resultData.check_out_time_formatted || nowFormatted,
        total_duration: resultData.total_duration || "00h 00m 00s"
      };
      setCaptureStatusText(`Identity verified (${candidateName}) — Check-Out Complete. Navigating to Inside page...`);

      setTimeout(() => {
        navigate("/inside", {
          state: {
            checkoutCompleted: true,
            checkoutData
          }
        });
      }, 1000);
    }
  };

  // Primary Two-Stage Identity Verification Workflow
  const runIdentityVerification = async () => {
    setLoading(true);
    setVerificationResult(null);
    setCameraStatus("normal");
    setCapturedFrames([]);
    capturedBlobsRef.current = [];
    capturedDataUrlsRef.current = [];

    // Case A: Diagnostic Test Image upload mode
    if (testImageFile) {
      setVerificationStatus("Identifying user...");
      setVerificationStep(1);
      setCaptureStatusText("Analyzing Test Image...");

      try {
        const formData = new FormData();
        formData.append("file", testImageFile, "probe_image.jpg");
        formData.append("action", action);
        formData.append("dry_run", isDiagnosticMode ? "true" : "false");

        setTimeout(() => {
          setVerificationStatus("Biometric verification...");
          setVerificationStep(2);
        }, 300);

        const response = await api.post("/user/verify_identity", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });

        const resData = response.data;
        setVerificationResult(resData);

        if (resData.final_decision === "VERIFIED") {
          setCameraStatus("success");
          setVerificationStatus("Identity verified");
          setVerificationStep(3);
          handleSuccessfulVerification(resData);
        } else {
          setCameraStatus("failed");
          setVerificationStatus("Identity verification failed");
          setVerificationStep(3);
          const failMsg = resData.failure_stage === "ACCESS_CONTROL"
            ? (resData.iris_stage?.decision_reason || "Access control rule violated")
            : (resData.failure_stage === "BIOMETRIC_VERIFICATION"
              ? (resData.iris_stage?.decision_reason === "USER_ALREADY_INSIDE"
                ? "User is already inside premises"
                : (resData.iris_stage?.decision_reason === "USER_NOT_INSIDE"
                  ? "User is not currently inside premises"
                  : "Biometric verification score below threshold"))
              : "Identity not recognized");
          setCaptureStatusText(`Verification Failed: ${failMsg}`);
        }
      } catch (err) {
        console.error("Verification error:", err);
        setCameraStatus("failed");
        setVerificationStatus("Identity verification failed");
        setVerificationStep(3);
        setCaptureStatusText("Verification Request Failed");
      }

      setLoading(false);
      return;
    }

    // Case B: Live Camera 5-Frame Capture Workflow
    if (!webcamRef.current) {
      alert("Unable to capture image from camera. Ensure camera is enabled.");
      setLoading(false);
      return;
    }

    try {
      for (let i = 0; i < 5; i++) {
        const pose = POSE_INSTRUCTIONS[i];
        setCaptureStatusText(pose.text);

        if (i > 0) {
          await delay(350);
        }

        let imageSrc = webcamRef.current ? webcamRef.current.getScreenshot() : null;
        let retries = 0;
        while (!imageSrc && retries < 3) {
          await delay(200);
          imageSrc = webcamRef.current ? webcamRef.current.getScreenshot() : null;
          retries++;
        }

        if (!imageSrc) {
          alert(`Frame ${i + 1} capture failed: Camera stream unreadable.`);
          setLoading(false);
          setCaptureStatusText("Capture Failed: Camera Stream Error");
          return;
        }

        const blob = await fetch(imageSrc).then((res) => res.blob());

        let annotatedDataUrl = imageSrc;
        let detectionPayload = { faceDetected: true, leftEyeDetected: true, rightEyeDetected: true };

        try {
          const annotationRes = await annotateCapturedImage(imageSrc);
          if (annotationRes && annotationRes.annotatedDataUrl) {
            annotatedDataUrl = annotationRes.annotatedDataUrl;
            detectionPayload = annotationRes.detectionPayload || detectionPayload;
          }
        } catch (annErr) {
          console.warn(`[Frame ${i + 1}] Annotation warning:`, annErr);
        }

        const frameObj = {
          rawSrc: imageSrc,
          annotatedSrc: annotatedDataUrl,
          poseLabel: pose.label,
          poseTag: pose.tag,
          ...detectionPayload
        };

        capturedBlobsRef.current.push(blob);
        capturedDataUrlsRef.current.push(frameObj);
        setCapturedFrames([...capturedDataUrlsRef.current]);
      }

      setCaptureStatusText("5/5 Frames Captured — Analyzing Biometric Data (Coordinated Request)...");
      setVerificationStatus("Identifying user...");
      setVerificationStep(1);

      // Coordinated Single Request for all 5 Captured Frames
      const coordinatedFormData = new FormData();
      capturedBlobsRef.current.forEach((blob, idx) => {
        coordinatedFormData.append("files", blob, `live_frame_${idx}.jpg`);
      });
      coordinatedFormData.append("action", action);
      coordinatedFormData.append("dry_run", isDiagnosticMode ? "true" : "false");

      setTimeout(() => {
        setVerificationStatus("Biometric verification...");
        setVerificationStep(2);
      }, 400);

      const resp = await api.post("/user/verify_identity", coordinatedFormData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      const finalRes = resp.data;

      setVerificationResult(finalRes);

      if (finalRes.final_decision === "VERIFIED") {
        setCameraStatus("success");
        setVerificationStatus("Identity verified");
        setVerificationStep(3);
        handleSuccessfulVerification(finalRes);
      } else {
        setCameraStatus("failed");
        setVerificationStatus("Identity verification failed");
        setVerificationStep(3);
        const failReason = finalRes.failure_stage === "ACCESS_CONTROL"
          ? (finalRes.iris_stage?.decision_reason || "Access control rule violated")
          : (finalRes.failure_stage === "BIOMETRIC_VERIFICATION"
            ? (finalRes.iris_stage?.decision_reason === "USER_ALREADY_INSIDE"
              ? "User is already inside premises"
              : (finalRes.iris_stage?.decision_reason === "USER_NOT_INSIDE"
                ? "User is not currently inside premises"
                : "Biometric verification score below threshold"))
            : "Identity not recognized");
        setCaptureStatusText(`Verification Failed: ${failReason}`);
      }
    } catch (error) {
      console.error("Two-Stage Verification Error:", error);
      setCameraStatus("failed");
      setVerificationStatus("Identity verification failed");
      setVerificationStep(3);
      setCaptureStatusText("Verification Error");
    }

    setLoading(false);
  };

  const currentCount = capturedFrames.length;

  return (
    <div className="verify-page-wrapper">
      <div className="verify-three-column-grid">
        {/* BLOCK 1: LIVE CAMERA & CONTROLS */}
        <div className="verify-card block-camera">
          <div className="card-header-title">🛡️ IDENTITY VERIFICATION</div>

          {/* ACCESS ACTION SELECTOR */}
          <div className="action-toggle-container">
            <label className="action-toggle-label">🚪 Select Intended Access Action:</label>
            <div className="action-btn-group">
              <button
                className={`action-toggle-btn btn-checkin ${action === "CHECK-IN" ? "active" : ""}`}
                onClick={() => setAction("CHECK-IN")}
                type="button"
                disabled={loading}
              >
                🟢 CHECK-IN
              </button>
              <button
                className={`action-toggle-btn btn-checkout ${action === "CHECK-OUT" ? "active" : ""}`}
                onClick={() => setAction("CHECK-OUT")}
                type="button"
                disabled={loading}
              >
                🔴 CHECK-OUT
              </button>
            </div>
          </div>

          {cameraDevices.length > 0 && (
            <div className="user-selector-container" style={{ marginTop: "4px", marginBottom: "12px" }}>
              <label>📷 Select Camera Device:</label>
              <select
                value={selectedDeviceId}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
                disabled={loading}
              >
                {cameraDevices.map((d, idx) => (
                  <option key={d.deviceId || idx} value={d.deviceId}>
                    {d.label || `Camera ${idx + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className={`camera-frame ${cameraStatus}`}>
            {testImagePreview ? (
              <img
                src={testImagePreview}
                alt="Diagnostic Test Preview"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            ) : (
              <>
                <Webcam
                  key={selectedDeviceId || "default"}
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  width={440}
                  height={320}
                  onUserMedia={refreshCameraDevices}
                  videoConstraints={
                    selectedDeviceId
                      ? { deviceId: { exact: selectedDeviceId }, width: 1280, height: 720 }
                      : { facingMode: "user" }
                  }
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="webcam-overlay-canvas"
                />
              </>
            )}
          </div>

          <div className="button-group" style={{ marginTop: "14px" }}>
            <button
              className="verify-btn"
              onClick={runIdentityVerification}
              disabled={loading}
              style={{
                background: "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)"
              }}
            >
              {loading
                ? (captureStatusText || "Processing Verification...")
                : "🛡️ START IDENTITY VERIFICATION"}
            </button>

            {(verificationResult || currentCount > 0 || testImageFile) && (
              <button
                className="btn-reset-cam"
                onClick={resetVerificationSequence}
                disabled={loading}
              >
                🔄 Reset Camera Stream
              </button>
            )}
          </div>

          {/* MANUAL ESP32 HARDWARE DIAGNOSTICS */}
          <div className="hw-test-card">
            <div className="hw-test-title">🧪 Manual ESP32 Hardware Diagnostics</div>
            <div className="hw-test-actions">
              <button
                type="button"
                className="hw-btn hw-btn-denied"
                onClick={() => handleManualHardwareTest("U")}
                disabled={hwTesting || loading}
              >
                {hwTesting ? "⏳ Sending..." : "🚨 TEST DENIED ALARM"}
              </button>
              <button
                type="button"
                className="hw-btn hw-btn-auth"
                onClick={() => handleManualHardwareTest("A")}
                disabled={hwTesting || loading}
              >
                {hwTesting ? "⏳ Sending..." : "✅ TEST AUTHORIZED INDICATOR"}
              </button>
            </div>
            {hwFeedback && (
              <div className={`hw-feedback-text ${hwFeedback.status === "success" ? "text-success" : (hwFeedback.status === "error" ? "text-danger" : "")}`}>
                {hwFeedback.message}
              </div>
            )}
          </div>
        </div>

        {/* BLOCK 2: CAPTURED FRAMES (5) */}
        <div className="verify-card block-captured">
          <div className="card-header-title">📷 CAPTURED FRAMES — {capturedFrames.length}/5</div>

          {captureStatusText ? (
            <div className="capture-status-badge">{captureStatusText}</div>
          ) : (
            <div className="capture-status-badge placeholder">Click 'Start Identity Verification' to begin capture</div>
          )}

          {capturedFrames.length === 0 ? (
            <div className="empty-captured-state">
              <div className="empty-icon">📷</div>
              <p>No frames captured yet.</p>
              <small>Click "🛡️ START IDENTITY VERIFICATION" to start guided 5-frame sequence.</small>
            </div>
          ) : (
            <div className="captured-grid">
              {capturedFrames.map((frameObj, index) => (
                <div key={index} className="frame-thumbnail-card">
                  <img
                    src={typeof frameObj === "string" ? frameObj : (frameObj.annotatedSrc || frameObj.rawSrc)}
                    alt={`Frame ${index + 1}`}
                    className="frame-thumb-img"
                  />
                  <div className="frame-label">{frameObj.poseLabel || `Frame ${index + 1}`}</div>
                  <div className="frame-status-tags-container">
                    <span className={`tag-pill ${typeof frameObj === "object" && frameObj.faceDetected ? "pass" : "fail"}`}>
                      {typeof frameObj === "object" && frameObj.faceDetected ? "✅ Face" : "❌ Face"}
                    </span>
                    <span className={`tag-pill ${typeof frameObj === "object" && frameObj.leftEyeDetected ? "pass" : "fail"}`}>
                      {typeof frameObj === "object" && frameObj.leftEyeDetected ? "✅ Left Eye" : "❌ Left Eye"}
                    </span>
                    <span className={`tag-pill ${typeof frameObj === "object" && frameObj.rightEyeDetected ? "pass" : "fail"}`}>
                      {typeof frameObj === "object" && frameObj.rightEyeDetected ? "✅ Right Eye" : "❌ Right Eye"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* BLOCK 3: VERIFICATION RESULT */}
        <div className="verify-card block-result">
          <div className="card-header-title">📊 VERIFICATION RESULT</div>

          {loading && !verificationResult ? (
            <div className="result-details" style={{ textAlign: "center", padding: "40px 10px" }}>
              <div className="result-status-banner">
                <span className="status-pill warning">
                  ⏳ {verificationStatus || "Identifying user..."}
                </span>
                <div className="decision-code">Stage {verificationStep} of 2</div>
              </div>
              <p style={{ color: "#64748b", fontSize: "14px", marginTop: "12px" }}>
                {verificationStep === 1 ? "Detecting face and identifying enrolled identity..." : "Performing candidate-specific biometric verification..."}
              </p>
            </div>
          ) : !verificationResult ? (
            <div className="empty-result-state">
              <div className="empty-icon">🛡️</div>
              <p>Awaiting Verification</p>
              <small>Initiate camera capture to run identity verification.</small>
            </div>
          ) : (
            <div className="result-details">
              <div className="result-status-banner">
                <span
                  className={`status-pill ${
                    verificationResult.final_decision === "VERIFIED"
                      ? "granted"
                      : "denied"
                  }`}
                >
                  {verificationResult.final_decision === "VERIFIED"
                    ? "🟢 Identity verified"
                    : "🔴 Identity verification failed"}
                </span>

                {verificationResult.final_decision === "VERIFIED" ? (
                  <div style={{ fontSize: "16px", fontWeight: "800", color: "#047857", marginTop: "6px" }}>
                    Identified User: {verificationResult.face_stage?.candidate_user_name}
                  </div>
                ) : (
                  <div className="decision-code" style={{ marginTop: "6px" }}>
                    Reason: <code>
                      {verificationResult.failure_stage === "BIOMETRIC_VERIFICATION"
                        ? "Biometric verification score below threshold"
                        : verificationResult.face_stage?.status === "MULTIPLE_FACES_DETECTED"
                        ? "Multiple faces detected"
                        : verificationResult.face_stage?.status === "NO_FACE_DETECTED"
                        ? "No face detected"
                        : "Identity not recognized"}
                    </code>
                  </div>
                )}
              </div>

              <div className="metrics-list">
                <div className="metric-row">
                  <span className="metric-label">Identified Candidate</span>
                  <span className="metric-val bold">
                    {verificationResult.face_stage?.candidate_user_name
                      ? `${verificationResult.face_stage.candidate_user_name} (#${verificationResult.face_stage.candidate_user_id})`
                      : "None"}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-label">Identity Identification</span>
                  <span className={`metric-val ${verificationResult.face_stage?.matched ? "text-success" : "text-danger"}`}>
                    {verificationResult.face_stage?.matched ? "Passed" : "Failed"}
                    {verificationResult.face_stage?.score !== undefined && ` (${verificationResult.face_stage.score.toFixed(4)})`}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-label">Biometric Verification</span>
                  <span className={`metric-val ${verificationResult.iris_stage?.matched ? "text-success" : "text-danger"}`}>
                    {verificationResult.iris_stage?.matched ? "Verified" : "Below Threshold"}
                    {verificationResult.iris_stage?.score !== undefined && ` (${verificationResult.iris_stage.score.toFixed(4)})`}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-label">Biometric Threshold</span>
                  <span className="metric-val">0.800</span>
                </div>

                {verificationResult.iris_stage?.template_ids_used && verificationResult.iris_stage.template_ids_used.length > 0 && (
                  <div className="metric-row">
                    <span className="metric-label">Templates Queried</span>
                    <span className="metric-val" style={{ fontSize: "12px" }}>
                      [{verificationResult.iris_stage.template_ids_used.join(", ")}]
                    </span>
                  </div>
                )}

                <div className="metric-row">
                  <span className="metric-label">Verification Mode</span>
                  <span className={`metric-val ${isDiagnosticMode ? "badge-off" : "badge-live"}`}>
                    {isDiagnosticMode ? "DIAGNOSTIC (SUPPRESSED)" : "LIVE VERIFICATION"}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-label">Hardware Status</span>
                  <span className={`metric-val ${isDiagnosticMode ? "badge-off" : "badge-live"}`}>
                    {isDiagnosticMode
                      ? "0 ACTIVATIONS (SAFETY ACTIVE)"
                      : (verificationResult.hardware_action || "HARDWARE_TRIGGERED")}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Verify;