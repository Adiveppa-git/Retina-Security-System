import { Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import Register from "./pages/Register";
import Verify from "./pages/Verify";
import Users from "./pages/Users";
import Logs from "./pages/Logs";
import Dashboard from "./pages/Dashboard";
import TimeStarter from "./pages/TimeStarter";
import Inside from "./pages/Inside";

import "./styles/App.css";

function App() {
  return (
    <div className="app-layout">
      <Navbar />
      <main className="app-main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/register" element={<Register />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/time-starter" element={<TimeStarter />} />
          <Route path="/inside" element={<Inside />} />
          <Route path="/users" element={<Users />} />
          <Route path="/logs" element={<Logs />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;