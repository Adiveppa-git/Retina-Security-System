import { Link } from "react-router-dom";
import "../styles/Home.css";

function Home() {
  return (
    <div className="home">

      <div className="hero">

        <h1>Iris Security System</h1>

        <h2>
          AI Based Smart Unauthorized Person Access Detection
        </h2>

        <p>
          Secure authentication using
          <strong> Anti-Spoofing, Iris Recognition and FastAPI.</strong>
        </p>

        <div className="buttons">

          <Link to="/register">
            <button>Register User</button>
          </Link>

          <Link to="/verify">
            <button>Verify Iris</button>
          </Link>

        </div>

      </div>

    </div>
  );
}

export default Home;