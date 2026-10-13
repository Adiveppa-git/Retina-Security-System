import { useState } from "react";
import { NavLink } from "react-router-dom";
import { FaShieldAlt, FaBars, FaTimes } from "react-icons/fa";
import "../styles/Navbar.css";

function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleMenu = () => setMobileMenuOpen(!mobileMenuOpen);
  const closeMenu = () => setMobileMenuOpen(false);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <NavLink to="/" className="navbar-brand" onClick={closeMenu}>
          <FaShieldAlt className="brand-icon" />
          <span>Iris Security System</span>
        </NavLink>

        <button className="mobile-toggle" onClick={toggleMenu} aria-label="Toggle Navigation Menu">
          {mobileMenuOpen ? <FaTimes /> : <FaBars />}
        </button>

        <ul className={`nav-links ${mobileMenuOpen ? "active" : ""}`}>
          <li>
            <NavLink
              to="/"
              end
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Home
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/register"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Register User
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/verify"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Verify Iris
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/users"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Users
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/inside"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              🟢 Inside
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/logs"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Access Logs
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/dashboard"
              onClick={closeMenu}
              className={({ isActive }) =>
                isActive ? "nav-link active-link" : "nav-link"
              }
            >
              Dashboard
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;