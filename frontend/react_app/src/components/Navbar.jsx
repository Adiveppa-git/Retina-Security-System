import { NavLink } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {

  return (

    <nav className="navbar">

      <div className="logo">
        Iris Security System
      </div>

      <ul className="nav-links">

        <li>
          <NavLink
            to="/"
            end
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
            className={({ isActive }) =>
              isActive ? "nav-link active-link" : "nav-link"
            }
          >
            Users
          </NavLink>
        </li>

        <li>
          <NavLink
            to="/logs"
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
            className={({ isActive }) =>
              isActive ? "nav-link active-link" : "nav-link"
            }
          >
            Dashboard
          </NavLink>
        </li>

      </ul>

    </nav>

  );

}

export default Navbar;