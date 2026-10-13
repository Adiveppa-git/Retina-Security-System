import { useEffect, useState } from "react";
import api from "../services/api";
import "../styles/Logs.css";

function Logs() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    try {
      const response = await api.get("/logs");
      setLogs(response.data);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="logs-container">
      <h1>System Access Logs</h1>

      <div className="logs-table-card">
        <table className="logs-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Time</th>
              <th>Person / Identity</th>
              <th>Similarity</th>
              <th>Status</th>
              <th>Response Time</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td colSpan="6" style={{ textAlign: "center", color: "#64748b", padding: "24px" }}>
                  No access logs recorded yet.
                </td>
              </tr>
            ) : (
              logs.map((log, index) => (
                <tr key={index}>
                  <td>{log.Date}</td>
                  <td>{log.Time}</td>
                  <td><strong>{log.Person}</strong></td>
                  <td>{log.Similarity}</td>
                  <td>
                    <span
                      className={`status-badge ${
                        log.Status === "ACCESS_GRANTED" || log.Status === "Success"
                          ? "granted"
                          : log.Status === "SPOOF_DETECTED" || log.Status === "Fake Iris"
                          ? "fake"
                          : "denied"
                      }`}
                    >
                      {log.Status === "ACCESS_GRANTED" || log.Status === "Success"
                        ? "🟢 ACCESS_GRANTED"
                        : log.Status === "SPOOF_DETECTED" || log.Status === "Fake Iris"
                        ? "⚠️ SPOOF_DETECTED"
                        : "🔴 ACCESS_DENIED"}
                    </span>
                  </td>
                  <td>{log["Response Time"]}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Logs;