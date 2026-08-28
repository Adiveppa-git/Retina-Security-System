import { useState } from "react";
import api from "../services/api";
import "../styles/Register.css";

function Register() {
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const registerUser = async (e) => {
    e.preventDefault();

    console.log("✅ Register button clicked");

    if (!name || !file) {
      setMessage("Please enter User ID and select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);

    try {
      console.log("Sending request...");

      const response = await api.post(
        "/user/add",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log("Backend Response:", response.data);

      if (response.data.error) {
        setMessage(response.data.error);
      } else {
        setMessage(
          `✅ User ${response.data.user} registered successfully.\nTotal Users: ${response.data.total_users}`
        );

        setName("");
        setFile(null);

        const fileInput = document.getElementById("file");
        if (fileInput) fileInput.value = "";
      }
    } catch (error) {
      console.error("Axios Error:", error);

      if (error.response) {
        console.log("Response:", error.response.data);
        setMessage(JSON.stringify(error.response.data));
      } else if (error.request) {
        console.log("Request:", error.request);
        setMessage("Network Error: Backend is unreachable.");
      } else {
        console.log("Message:", error.message);
        setMessage(error.message);
      }
    }
  };

  return (
    <div className="register-container">
      <form className="register-form" onSubmit={registerUser}>
        <h2>Register New User</h2>

        <input
          type="text"
          placeholder="User ID (Example: 021)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <input
          id="file"
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button type="submit">
          Register User
        </button>

        <p style={{ whiteSpace: "pre-line" }}>
          {message}
        </p>
      </form>
    </div>
  );
}

export default Register;