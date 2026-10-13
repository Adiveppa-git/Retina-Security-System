import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import requests
import time

BASE_URL = "http://127.0.0.1:8000"

print("--- TESTING TIME STARTER & AUTOMATIC NAVIGATION BACKEND FLOW ---")

# 1. Fetch user
users = requests.get(f"{BASE_URL}/user/all").json()
if not users:
    print("No users found.")
    sys.exit(1)

test_user = users[0]
print(f"Testing with User '{test_user['name']}' (#{test_user['id']})...")

from app.database.connection import SessionLocal
from app.models.user_model import User, AccessLog
from app.services.iris_recognition_service import log_access_event

db = SessionLocal()

# 2. Simulate CHECK-IN Verification event
print("\n1. Simulating Genuine CHECK-IN Verification Event...")
log1 = log_access_event(db, test_user['name'], "ACCESS_GRANTED", 0.9512, action="CHECK-IN")
checkin_dt = log1.timestamp
print("   Check-In Logged! Timestamp:", checkin_dt.isoformat())

# 3. Test Active Session Recovery endpoint GET /iris/active_session
print("\n2. Testing Active Session Recovery (GET /iris/active_session)...")
active_res = requests.get(f"{BASE_URL}/iris/active_session/{test_user['id']}").json()
print("   Active Session Response:", active_res)
assert active_res["active"] == True, "Active session recovery failed!"
assert active_res["user_name"] == test_user['name'], "User mismatch!"
print("   ✓ Active session successfully recovered!")

# 4. Wait 3 seconds to simulate time inside
print("\n3. Waiting 3 seconds inside premises...")
time.sleep(3)

# 5. Simulate CHECK-OUT Verification event & calculate total duration
print("\n4. Simulating Genuine CHECK-OUT Verification Event...")
log2 = log_access_event(db, test_user['name'], "ACCESS_GRANTED", 0.9388, action="CHECK-OUT")
checkout_dt = log2.timestamp
diff_sec = int((checkout_dt - checkin_dt).total_seconds())
hrs = diff_sec // 3600
mins = (diff_sec % 3600) // 60
secs = diff_sec % 60
duration_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"

print("   Check-Out Logged! Timestamp:", checkout_dt.isoformat())
print("   Calculated Total Duration:", duration_str)

# 6. Verify Active Session is now inactive (User checked out)
print("\n5. Verifying Active Session after Check-Out...")
post_out_res = requests.get(f"{BASE_URL}/iris/active_session/{test_user['id']}").json()
print("   Post-Checkout Session Response:", post_out_res)
assert post_out_res["active"] == False, "Session should be inactive post check-out!"
print("   ✓ Session correctly marked INACTIVE post check-out!")

db.close()
print("\n--- ALL TESTS COMPLETED SUCCESSFULLY ---")
