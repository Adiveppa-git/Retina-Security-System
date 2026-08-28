import { Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import Register from "./pages/Register";
import Verify from "./pages/Verify";
import Users from "./pages/Users";
import Logs from "./pages/Logs";
import Dashboard from "./pages/Dashboard";

import "./styles/App.css";

function App() {
  return (
    <>
      <Navbar />

      <Routes>

        <Route path="/" element={<Home />} />

        <Route path="/register" element={<Register />} />

        <Route path="/verify" element={<Verify />} />

        <Route path="/users" element={<Users />} />

        <Route path="/logs" element={<Logs />} />

        <Route path="/dashboard" element={<Dashboard />} />

      </Routes>

    </>
  );
}

export default App;