import { useEffect, useState } from "react";
import api from "../services/api";
import "../styles/Dashboard.css";
import DashboardCharts from "../components/DashboardCharts";

import {
  FaUsers,
  FaCheckCircle,
  FaTimesCircle,
  FaChartLine,
  FaClock,
  FaDatabase,
  FaShieldAlt
} from "react-icons/fa";

function Dashboard() {

  const [data, setData] = useState(null);
  const [time, setTime] = useState(new Date());
  const [loading, setLoading] = useState(true);

  // ==========================================
  // Load Dashboard
  // ==========================================

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

  // ==========================================
  // Auto Refresh
  // ==========================================

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

  // ==========================================
  // Loading Screen
  // ==========================================

  if (loading || !data) {

    return (

      <div className="loading">

        <FaShieldAlt className="loading-icon" />

        <h2>Loading Dashboard...</h2>

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

            <h1>

              AI Iris Security Dashboard

            </h1>

            <p>

              Smart Biometric Authentication & Monitoring

            </p>

          </div>

        </div>

        <div className="header-right">

          <div className="clock-box">

            <h2>

              {time.toLocaleTimeString()}

            </h2>

            <p>

              {time.toDateString()}

            </p>

          </div>

        </div>

      </div>

      {/* ==========================================
          STATISTICS
      ========================================== */}

      <div className="card-grid">

        <div className="dashboard-card users">

          <FaUsers className="card-icon" />

          <h2>{data.total_users}</h2>

          <p>Total Users</p>

        </div>

        <div className="dashboard-card requests">

          <FaDatabase className="card-icon" />

          <h2>{data.total_requests}</h2>

          <p>Total Requests</p>

        </div>

        <div className="dashboard-card success">

          <FaCheckCircle className="card-icon" />

          <h2>{data.authorized}</h2>

          <p>Authorized</p>

        </div>

        <div className="dashboard-card denied">

          <FaTimesCircle className="card-icon" />

          <h2>{data.denied}</h2>

          <p>Denied</p>

        </div>

        <div className="dashboard-card rate">

          <FaChartLine className="card-icon" />

          <h2>{data.success_rate}</h2>

          <p>Success Rate</p>

        </div>

        <div className="dashboard-card response">

          <FaClock className="card-icon" />

          <h2>{data.average_response_time}</h2>

          <p>Average Response</p>

        </div>

      </div>

      {/* ==========================================
          CHARTS
      ========================================== */}

      <DashboardCharts data={data} />

      {/* ==========================================
          RECENT LOGS
      ========================================== */}

      <h2 className="recent-title">

        Recent Access Logs

      </h2>

      <div className="table-container">

        <table>

          <thead>

            <tr>

              <th>Date</th>

              <th>Time</th>

              <th>Person</th>

              <th>Status</th>

              <th>Similarity</th>

            </tr>

          </thead>

          <tbody>

            {data.recent_logs.length === 0 ? (

              <tr>

                <td colSpan="5">

                  No Access Logs Found

                </td>

              </tr>

            ) : (

              data.recent_logs.map((log, index) => (

                <tr key={index}>

                  <td>{log.Date}</td>

                  <td>{log.Time}</td>

                  <td>{log.Person}</td>

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

                  <td>{log.Similarity}</td>

                </tr>

              ))

            )}

          </tbody>

        </table>

      </div>

    </div>

  );

}

export default Dashboard;