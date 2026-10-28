import { createContext, useContext, useState, useEffect, useCallback } from "react";
import api from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem("irisguard_user");
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return Boolean(localStorage.getItem("irisguard_token") || localStorage.getItem("irisguard_user"));
  });

  const [loading, setLoading] = useState(true);

  // Validate session on app initialization
  const checkAuth = useCallback(async () => {
    try {
      const res = await api.get("/auth/me");
      if (res.data && res.data.authenticated) {
        setIsAuthenticated(true);
        setUser((prev) => prev || { username: res.data.username });
      } else {
        setIsAuthenticated(false);
        setUser(null);
        localStorage.removeItem("irisguard_token");
        localStorage.removeItem("irisguard_user");
      }
    } catch {
      // If 401 or network error, mark as unauthenticated
      setIsAuthenticated(false);
      setUser(null);
      localStorage.removeItem("irisguard_token");
      localStorage.removeItem("irisguard_user");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  const login = async (username, password) => {
    const res = await api.post("/auth/login", { username, password });
    if (res.data && res.data.token) {
      localStorage.setItem("irisguard_token", res.data.token);
      const userData = {
        username: res.data.username,
        full_name: res.data.full_name || "Administrator",
      };
      localStorage.setItem("irisguard_user", JSON.stringify(userData));
      setUser(userData);
      setIsAuthenticated(true);
      return res.data;
    }
    throw new Error(res.data?.detail || "Authentication failed");
  };

  const logout = async () => {
    try {
      await api.post("/auth/logout");
    } catch {
      // Invalidate client side even if server call errors
    } finally {
      localStorage.removeItem("irisguard_token");
      localStorage.removeItem("irisguard_user");
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        user,
        loading,
        login,
        logout,
        checkAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
