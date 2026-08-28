import { useRef, useState } from "react";
import Webcam from "react-webcam";
import api from "../services/api";
import "../styles/Verify.css";

function Verify() {

  const webcamRef = useRef(null);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [detected, setDetected] = useState(false);
  // webcam border status
  const [cameraStatus, setCameraStatus] = useState("normal");
  // normal | success | failed

  const captureAndVerify = async () => {

    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) {
      alert("Unable to capture image.");
      return;
    }

    setLoading(true);

    setCameraStatus("normal");

    try {

      const blob = await fetch(imageSrc).then((res) => res.blob());

      const formData = new FormData();

      formData.append(
        "file",
        blob,
        "capture.jpg"
      );

      const response = await api.post(
        "/iris/verify",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

     setResult(response.data);

if (
    response.data.status === "Success" ||
    response.data.status === "Access Granted"
) {

    setDetected(true);
    setCameraStatus("success");

} else {

    setDetected(false);
    setCameraStatus("failed");

}

    } catch (error) {

      console.log(error);

      alert("Verification Failed.");

      setCameraStatus("failed");

    }

    setLoading(false);

  };

  return (

    <div className="verify-container">

      <div className="verify-card">

        <h2>Live Iris Verification</h2>

        <div className={`camera-frame ${cameraStatus}`}>

      <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={420}
          videoConstraints={{
              facingMode:"user"
          }}
      />

  </div>

        <button
          className="verify-btn"
          onClick={captureAndVerify}
          disabled={loading}
        >

          {loading ? "Verifying..." : "Capture & Verify"}

        </button>

      </div>

      {result && (

        <div className="result-card">

          <h2>Verification Result</h2>

          <p>
            <strong>Status :</strong> {result.status}
          </p>

          <p>
            <strong>Person :</strong> {result.person}
          </p>

          <p>
            <strong>Similarity :</strong> {result.similarity}
          </p>

          <p>
            <strong>Response :</strong> {result.response_time}
          </p>

          <hr />

          <h3>Hardware Status</h3>

          <p>
            <strong>Door :</strong> {result.hardware.door}
          </p>

          <p>
            <strong>LED :</strong> {result.hardware.led}
          </p>

          <p>
            <strong>Buzzer :</strong> {result.hardware.buzzer}
          </p>

        </div>

      )}

    </div>

  );

}

export default Verify;