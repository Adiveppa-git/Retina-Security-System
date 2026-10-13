import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import requests

BASE_URL = "http://127.0.0.1:8000"

print("--- TESTING ENTRY / EXIT VERIFICATION & DASHBOARD API ---")

# 1. Fetch enrolled users
users_resp = requests.get(f"{BASE_URL}/user/all")
print("Enrolled Users HTTP Status:", users_resp.status_code)
users = users_resp.json()
print("Found", len(users), "users in database.")

if users:
    test_user = users[0]
    print(f"Testing with User '{test_user['name']}' (#{test_user['id']})...")

    # 2. Test Dashboard before verification
    dash1 = requests.get(f"{BASE_URL}/dashboard/").json()
    print("\nInitial Dashboard State:")
    print("  - Total Users:", dash1.get("total_users"))
    print("  - Currently Inside Count:", dash1.get("currently_inside_count"))
    print("  - Check-ins Today:", dash1.get("checkins_today"))
    print("  - Check-outs Today:", dash1.get("checkouts_today"))

print("\n--- TEST COMPLETE ---")
