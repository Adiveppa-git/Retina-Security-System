import requests

BASE_URL = "http://127.0.0.1:8000"

print("--- TESTING /inside BACKEND API ---")
res = requests.get(f"{BASE_URL}/iris/active_sessions").json()
print("Response:", res)
assert "active_users" in res
print("✓ API test passed successfully!")
