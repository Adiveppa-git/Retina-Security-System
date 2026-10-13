import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import requests
import io
import time
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

print("--- EXECUTING ALL 8 MANDATORY ACCESS CONTROL TESTS ---")

# Fetch user
users = requests.get(f"{BASE_URL}/user/all").json()
if not users:
    print("No users found in DB.")
    exit(1)

test_user = users[0]
user_id = test_user["id"]
user_name = test_user["name"]
print(f"Testing with User '{user_name}' (#{user_id})...\n")

from app.database.connection import SessionLocal
from app.models.user_model import AccessLog
from app.services.iris_recognition_service import log_access_event, recognize_iris_v2_1to1_diagnostic

# 0. Reset state: Ensure user starts OUTSIDE premises
db = SessionLocal()
last_granted = db.query(AccessLog).filter(AccessLog.user_name == user_name, AccessLog.status == "ACCESS_GRANTED").order_by(AccessLog.timestamp.desc()).first()
if last_granted and last_granted.action == "CHECK-IN":
    log_access_event(db, user_name, "ACCESS_GRANTED", 0.90, action="CHECK-OUT")
db.close()

# TEST 1: Registered user + successful CHECK-IN -> user appears on /inside
print("[TEST 1] Successful CHECK-IN for user...")
db1 = SessionLocal()
log1 = log_access_event(db1, user_name, "ACCESS_GRANTED", 0.92, action="CHECK-IN")
db1.close()

active_sessions = requests.get(f"{BASE_URL}/iris/active_sessions").json()
active_names = [u["user_name"] for u in active_sessions.get("active_users", [])]
print("   Active Users on /inside:", active_names)
assert user_name in active_names, "TEST 1 FAILED: User should be inside!"
print("   [PASSED] TEST 1: User successfully checked in and appears on /inside")

# TEST 2: Same user attempts CHECK-IN again -> DENIED (USER_ALREADY_INSIDE)
print("\n[TEST 2] Duplicate CHECK-IN attempt for same user...")
img = Image.new("RGB", (320, 240), color=(100, 100, 100))
res_dup = recognize_iris_v2_1to1_diagnostic(img, user_id, action="CHECK-IN")
print("   Duplicate Check-in Status:", res_dup["final_diagnostic_status"])
print("   Duplicate Check-in Reason:", res_dup["decision_reason"])
assert res_dup["decision_reason"] == "USER_ALREADY_INSIDE", "TEST 2 FAILED: Duplicate check-in should be rejected!"
print("   [PASSED] TEST 2: Duplicate check-in correctly rejected with USER_ALREADY_INSIDE")

# TEST 3: User performs CHECK-OUT -> user disappears from /inside
print("\n[TEST 3] Successful CHECK-OUT for user...")
db3 = SessionLocal()
log2 = log_access_event(db3, user_name, "ACCESS_GRANTED", 0.91, action="CHECK-OUT")
db3.close()

active_sessions_post_out = requests.get(f"{BASE_URL}/iris/active_sessions").json()
active_names_post = [u["user_name"] for u in active_sessions_post_out.get("active_users", [])]
print("   Active Users on /inside post-checkout:", active_names_post)
assert user_name not in active_names_post, "TEST 3 FAILED: User should be removed from /inside!"
print("   [PASSED] TEST 3: User checked out and disappeared from /inside")

# TEST 4: Same user performs CHECK-IN again -> SUCCESS
print("\n[TEST 4] Re-checking in same user after check-out...")
db4 = SessionLocal()
log3 = log_access_event(db4, user_name, "ACCESS_GRANTED", 0.93, action="CHECK-IN")
db4.close()

active_sessions_re = requests.get(f"{BASE_URL}/iris/active_sessions").json()
active_names_re = [u["user_name"] for u in active_sessions_re.get("active_users", [])]
assert user_name in active_names_re, "TEST 4 FAILED: User should be able to check in again!"
print("   [PASSED] TEST 4: User re-checked in successfully")

# Clean check-out for next tests
db_clean = SessionLocal()
log_access_event(db_clean, user_name, "ACCESS_GRANTED", 0.90, action="CHECK-OUT")
db_clean.close()

# TEST 5: Unverified / Rejected person attempts CHECK-IN -> DENIED, no check-in record
print("\n[TEST 5] Unverified/rejected verification attempt...")
db5 = SessionLocal()
log_fail = log_access_event(db5, user_name, "ACCESS_DENIED", 0.45, action="CHECK-IN")
db5.close()

active_sessions_fail = requests.get(f"{BASE_URL}/iris/active_sessions").json()
active_names_fail = [u["user_name"] for u in active_sessions_fail.get("active_users", [])]
assert user_name not in active_names_fail, "TEST 5 FAILED: Failed verification must NOT grant entry!"
print("   [PASSED] TEST 5: Failed verification denied entry and created no active session")

# TEST 6: User attempts CHECK-OUT while not inside -> DENIED (USER_NOT_INSIDE)
print("\n[TEST 6] CHECK-OUT attempt when user is NOT inside...")
res_out_fail = recognize_iris_v2_1to1_diagnostic(img, user_id, action="CHECK-OUT")
print("   Check-Out while Outside Status:", res_out_fail["final_diagnostic_status"])
print("   Check-Out while Outside Reason:", res_out_fail["decision_reason"])
assert res_out_fail["decision_reason"] == "USER_NOT_INSIDE", "TEST 6 FAILED: Check-out while outside should be rejected!"
print("   [PASSED] TEST 6: Check-out while outside correctly rejected with USER_NOT_INSIDE")

# TEST 7: Refresh /inside API -> only genuinely active sessions shown
print("\n[TEST 7] Refresh /inside API...")
db7 = SessionLocal()
log_valid = log_access_event(db7, user_name, "ACCESS_GRANTED", 0.94, action="CHECK-IN")
db7.close()

refreshed = requests.get(f"{BASE_URL}/iris/active_sessions").json()
refreshed_users = refreshed.get("active_users", [])
assert len(refreshed_users) == 1 and refreshed_users[0]["user_name"] == user_name
print("   [PASSED] TEST 7: /inside API returns only genuinely active check-in sessions")

# TEST 8: Direct API call for CHECK-IN while user is inside -> backend rejects it
print("\n[TEST 8] Direct API call for CHECK-IN while user is already inside...")
res_api_dup = recognize_iris_v2_1to1_diagnostic(img, user_id, action="CHECK-IN")
assert res_api_dup["decision_reason"] == "USER_ALREADY_INSIDE", "TEST 8 FAILED: Direct backend API must enforce rules!"
print("   [PASSED] TEST 8: Backend API directly rejects duplicate check-in")

# Clean up test session
db_end = SessionLocal()
log_access_event(db_end, user_name, "ACCESS_GRANTED", 0.90, action="CHECK-OUT")
db_end.close()

print("\n==========================================")
print("ALL 8 MANDATORY ACCESS CONTROL TESTS PASSED SUCCESSFULLY!")
print("==========================================")
