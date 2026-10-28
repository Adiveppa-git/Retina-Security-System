import sys, os
sys.path.insert(0, os.path.abspath("backend"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("--- 1. Testing Unauthenticated Request to Protected Route /dashboard/ ---")
res = client.get("/dashboard/")
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 401, f"Expected 401, got {res.status_code}"

print("\n--- 2. Testing Unauthenticated Request to /logs/ ---")
res = client.get("/logs/")
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 401

print("\n--- 3. Testing Failed Login (Wrong Password) ---")
res = client.post("/auth/login", json={"username": "admin", "password": "wrongpassword"})
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 401

print("\n--- 4. Testing Successful Login (admin / admin123) ---")
res = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 200
data = res.json()
token = data["token"]
cookie_val = res.cookies.get("irisguard_session")
print("Session token:", token)
print("Cookie value:", cookie_val)

print("\n--- 5. Testing /auth/me with Cookie ---")
client.cookies.set("irisguard_session", token)
res = client.get("/auth/me")
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 200
assert res.json()["username"] == "admin"

print("\n--- 6. Testing Protected Route /dashboard/ with Cookie ---")
res = client.get("/dashboard/")
print("Status code:", res.status_code)
assert res.status_code == 200

print("\n--- 7. Testing Protected Route with Bearer Token Header ---")
client_no_cookie = TestClient(app)
res = client_no_cookie.get("/dashboard/", headers={"Authorization": f"Bearer {token}"})
print("Status code with Bearer header:", res.status_code)
assert res.status_code == 200

print("\n--- 8. Testing Logout ---")
res = client.post("/auth/logout")
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 200

print("\n--- 9. Testing Protected Route After Logout ---")
res = client.get("/dashboard/")
print("Status code after logout:", res.status_code, "Body:", res.json())
assert res.status_code == 401

print("\n--- 10. Testing Login with Basu Account (basu / Basu@123) ---")
res = client.post("/auth/login", json={"username": "basu", "password": "Basu@123"})
print("Status code:", res.status_code, "Body:", res.json())
assert res.status_code == 200

print("\nALL BACKEND AUTH TESTS PASSED 100%!")
