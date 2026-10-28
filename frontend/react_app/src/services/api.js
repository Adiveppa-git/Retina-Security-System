import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  withCredentials: true, // Send HTTP-only session cookies automatically
});

// Attach Authorization Bearer token if stored locally for dual-mode support
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("irisguard_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: handle 401 Unauthorized responses safely
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear client-side stored session information
      localStorage.removeItem("irisguard_token");
      localStorage.removeItem("irisguard_user");

      // Redirect unauthenticated user to /login unless already on /login
      if (
        typeof window !== "undefined" &&
        !window.location.pathname.startsWith("/login")
      ) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

export default api;