import requests

BASE_URL = "http://127.0.0.1:8000"

print("================================================================")
print(" LIVE END-TO-END HTTP AUTHENTICATION & ROUTE PROTECTION TEST ")
print("================================================================")

# 1. Unauthenticated direct access to protected endpoints
print("\n[TEST 1] Testing direct unauthenticated access to /dashboard/...")
r1 = requests.get(f"{BASE_URL}/dashboard/")
print(f"Status: {r1.status_code} | Body: {r1.text}")
assert r1.status_code == 401, f"Expected 401, got {r1.status_code}"

print("\n[TEST 2] Testing direct unauthenticated access to /user/all...")
r2 = requests.get(f"{BASE_URL}/user/all")
print(f"Status: {r2.status_code} | Body: {r2.text}")
assert r2.status_code == 401, f"Expected 401, got {r2.status_code}"

print("\n[TEST 3] Testing direct unauthenticated access to /logs/...")
r3 = requests.get(f"{BASE_URL}/logs/")
print(f"Status: {r3.status_code} | Body: {r3.text}")
assert r3.status_code == 401, f"Expected 401, got {r3.status_code}"

# 2. Failed login attempt
print("\n[TEST 4] Testing failed login with invalid password...")
r4 = requests.post(f"{BASE_URL}/auth/login", json={"username": "admin", "password": "wrong_password_xyz"})
print(f"Status: {r4.status_code} | Body: {r4.text}")
assert r4.status_code == 401, f"Expected 401, got {r4.status_code}"

# 3. Successful login with existing admin account
print("\n[TEST 5] Testing successful login with 'admin' / 'admin123'...")
session = requests.Session()
r5 = session.post(f"{BASE_URL}/auth/login", json={"username": "admin", "password": "admin123"})
print(f"Status: {r5.status_code} | Body: {r5.text}")
assert r5.status_code == 200, f"Expected 200, got {r5.status_code}"
login_data = r5.json()
token = login_data["token"]
cookie_val = session.cookies.get("irisguard_session")
print(f"Token: {token} | Session Cookie: {cookie_val}")
assert token is not None and len(token) > 20
assert cookie_val == token

# 4. Check /auth/me with session
print("\n[TEST 6] Testing /auth/me with active session...")
r6 = session.get(f"{BASE_URL}/auth/me")
print(f"Status: {r6.status_code} | Body: {r6.text}")
assert r6.status_code == 200
assert r6.json().get("username") == "admin"

# 5. Access protected endpoints with active session
print("\n[TEST 7] Testing access to /dashboard/ with active session...")
r7 = session.get(f"{BASE_URL}/dashboard/")
print(f"Status: {r7.status_code} | Dashboard keys: {list(r7.json().keys()) if r7.status_code == 200 else r7.text}")
assert r7.status_code == 200

print("\n[TEST 8] Testing access to /user/all with active session...")
r8 = session.get(f"{BASE_URL}/user/all")
print(f"Status: {r8.status_code} | Users count: {len(r8.json()) if r8.status_code == 200 else r8.text}")
assert r8.status_code == 200

print("\n[TEST 9] Testing access to /logs/ with active session...")
r9 = session.get(f"{BASE_URL}/logs/")
print(f"Status: {r9.status_code} | Logs count: {len(r9.json()) if r9.status_code == 200 else r9.text}")
assert r9.status_code == 200

# 6. Test Bearer Token header authentication (without cookies)
print("\n[TEST 10] Testing access using Authorization: Bearer <token> header...")
r10 = requests.get(f"{BASE_URL}/dashboard/", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r10.status_code}")
assert r10.status_code == 200

# 7. Logout and session invalidation
print("\n[TEST 11] Testing logout and session invalidation...")
r11 = session.post(f"{BASE_URL}/auth/logout")
print(f"Status: {r11.status_code} | Body: {r11.text}")
assert r11.status_code == 200

print("\n[TEST 12] Testing access to /dashboard/ after logout...")
r12 = session.get(f"{BASE_URL}/dashboard/")
print(f"Status: {r12.status_code} | Body: {r12.text}")
assert r12.status_code == 401

print("\n[TEST 13] Testing access with previously valid Bearer token after logout...")
r13 = requests.get(f"{BASE_URL}/dashboard/", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r13.status_code} | Body: {r13.text}")
assert r13.status_code == 401

# 8. Testing Basu account credentials
print("\n[TEST 14] Testing login with 'basu' / 'Basu@123'...")
session2 = requests.Session()
r14 = session2.post(f"{BASE_URL}/auth/login", json={"username": "basu", "password": "Basu@123"})
print(f"Status: {r14.status_code} | Body: {r14.text}")
assert r14.status_code == 200
assert session2.get(f"{BASE_URL}/dashboard/").status_code == 200

print("\n================================================================")
print(" ALL 14 LIVE END-TO-END TESTS PASSED WITH 100% SUCCESS! ")
print("================================================================")
