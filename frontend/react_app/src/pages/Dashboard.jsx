import { useEffect, useState } from "react";
import api from "../services/api";
import "../styles/Dashboard.css";
import DashboardCharts from "../components/DashboardCharts";

import {
  FaUsers,
  FaDoorOpen,
  FaSignInAlt,
  FaSignOutAlt,
  FaCheckCircle,
  FaTimesCircle,
  FaShieldAlt,
  FaHistory
} from "react-icons/fa";

function Dashboard() {
  const [data, setData] = useState(null);
  const [time, setTime] = useState(new Date());
  const [loading, setLoading] = useState(true);

  const loadDashboard = async () => {
    try {
      const response = await api.get("/dashboard/");
      setData(response.data);
    } catch (err) {
      console.log("Dashboard Error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();

    const clock = setInterval(() => {
      setTime(new Date());
    }, 1000);

    const refresh = setInterval(() => {
      loadDashboard();
    }, 10000);

    return () => {
      clearInterval(clock);
      clearInterval(refresh);
    };
  }, []);

  if (loading || !data) {
    return (
      <div className="loading">
        <FaShieldAlt className="loading-icon" />
        <h2>Loading Entry & Exit Security Dashboard...</h2>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* ==========================================
          HEADER
      ========================================== */}
      <div className="dashboard-header">
        <div className="header-left">
          <div className="shield-box">
            <FaShieldAlt className="shield-icon" />
          </div>
          <div>
            <h1>NEXGUARD</h1>
            <p>Intelligent Iris Security and Access Control</p>
          </div>
        </div>

        <div className="header-right">
          <div className="clock-box">
            <h2>{time.toLocaleTimeString()}</h2>
            <p>{time.toDateString()}</p>
          </div>
        </div>
      </div>

      {/* ==========================================
          REAL MYSQL STATISTICS CARDS (6 CARDS)
      ========================================== */}
      <div className="card-grid">
        <div className="dashboard-card users">
          <FaUsers className="card-icon" />
          <h2>{data.total_users}</h2>
          <p>Total Registered Users</p>
        </div>

        <div className="dashboard-card inside">
          <FaDoorOpen className="card-icon" />
          <h2>{data.currently_inside_count}</h2>
          <p>Currently Inside</p>
        </div>

        <div className="dashboard-card checkin">
          <FaSignInAlt className="card-icon" />
          <h2>{data.checkins_today}</h2>
          <p>Check-ins Today</p>
        </div>

        <div className="dashboard-card checkout">
          <FaSignOutAlt className="card-icon" />
          <h2>{data.checkouts_today}</h2>
          <p>Check-outs Today</p>
        </div>

        <div className="dashboard-card success">
          <FaCheckCircle className="card-icon" />
          <h2>{data.successful_today}</h2>
          <p>Successful Verifications</p>
        </div>

        <div className="dashboard-card denied">
          <FaTimesCircle className="card-icon" />
          <h2>{data.denied_today}</h2>
          <p>Denied Attempts</p>
        </div>
      </div>

      {/* ==========================================
          CURRENTLY INSIDE TABLE
      ========================================== */}
      <div className="dashboard-section-card">
        <div className="section-card-header">
          <h2>🟢 CURRENTLY INSIDE PREMISES</h2>
          <span className="count-badge">{data.currently_inside ? data.currently_inside.length : 0} Persons</span>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>User Name</th>
                <th>User ID</th>
                <th>Check-in Time</th>
                <th>Current Status</th>
              </tr>
            </thead>
            <tbody>
              {!data.currently_inside || data.currently_inside.length === 0 ? (
                <tr>
                  <td colSpan="4" className="empty-table-msg">
                    No users currently inside the premises.
                  </td>
                </tr>
              ) : (
                data.currently_inside.map((item, idx) => (
                  <tr key={idx}>
                    <td className="font-bold">{item.user_name}</td>
                    <td>#{item.user_id}</td>
                    <td>{item.checkin_time}</td>
                    <td>
                      <span className="badge inside-badge">CURRENTLY INSIDE</span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ==========================================
          COMPLETED VISITS & DURATION HISTORY
      ========================================== */}
      {data.completed_visits && data.completed_visits.length > 0 && (
        <div className="dashboard-section-card">
          <div className="section-card-header">
            <h2>⏱️ COMPLETED VISITS & DURATION HISTORY</h2>
            <span className="count-badge">{data.completed_visits.length} Visits</span>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>User Name</th>
                  <th>Check-In Time</th>
                  <th>Check-Out Time</th>
                  <th>Total Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {data.completed_visits.map((visit, idx) => (
                  <tr key={idx}>
                    <td className="font-bold">{visit.user_name}</td>
                    <td>{visit.check_in_formatted || "N/A"}</td>
                    <td>{visit.check_out_formatted || "N/A"}</td>
                    <td className="font-mono font-bold" style={{ color: "#38bdf8" }}>{visit.duration}</td>
                    <td>
                      <span className="badge action-badge checkout">COMPLETED</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ==========================================
          TODAY'S ACCESS LOG TABLE
      ========================================== */}
      <div className="dashboard-section-card">
        <div className="section-card-header">
          <h2>📋 TODAY'S ACCESS LOGS</h2>
          <span className="count-badge">{data.todays_access_logs ? data.todays_access_logs.length : 0} Logs</span>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>User Name</th>
                <th>Date</th>
                <th>Time</th>
                <th>Action</th>
                <th>Verification Result</th>
                <th>Similarity Score</th>
              </tr>
            </thead>
            <tbody>
              {!data.todays_access_logs || data.todays_access_logs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="empty-table-msg">
                    No access events recorded today.
                  </td>
                </tr>
              ) : (
                data.todays_access_logs.map((log, idx) => (
                  <tr key={idx}>
                    <td className="font-bold">{log.user_name}</td>
                    <td>{log.date}</td>
                    <td>{log.time}</td>
                    <td>
                      <span className={`badge action-badge ${log.action === "CHECK-OUT" ? "checkout" : "checkin"}`}>
                        {log.action}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${log.status === "ACCESS_GRANTED" ? "success-badge" : "denied-badge"}`}>
                        {log.status === "ACCESS_GRANTED" ? "ACCESS GRANTED" : "ACCESS DENIED"}
                      </span>
                    </td>
                    <td className="font-mono">{log.similarity_score}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ==========================================
          CHARTS
      ========================================== */}
      <DashboardCharts data={data} />

      {/* ==========================================
          RECENT ACTIVITY TABLE
      ========================================== */}
      <div className="dashboard-section-card">
        <div className="section-card-header">
          <h2><FaHistory style={{ marginRight: "8px" }} /> RECENT LOGS ACTIVITY</h2>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Time</th>
                <th>Person</th>
                <th>Action</th>
                <th>Status</th>
                <th>Similarity Score</th>
              </tr>
            </thead>
            <tbody>
              {!data.recent_logs || data.recent_logs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="empty-table-msg">
                    No recent access logs found in MySQL database.
                  </td>
                </tr>
              ) : (
                data.recent_logs.map((log, index) => (
                  <tr key={index}>
                    <td>{log.Date}</td>
                    <td>{log.Time}</td>
                    <td className="font-bold">{log.Person}</td>
                    <td>
                      <span className={`badge action-badge ${log.Action === "CHECK-OUT" ? "checkout" : "checkin"}`}>
                        {log.Action || "CHECK-IN"}
                      </span>
                    </td>
                    <td>
                      <span
                        className={
                          log.Status === "Success"
                            ? "badge success-badge"
                            : log.Status === "Fake Iris"
                              ? "badge fake-badge"
                              : "badge denied-badge"
                        }
                      >
                        {log.Status}
                      </span>
                    </td>
                    <td className="font-mono">{log.Similarity}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;