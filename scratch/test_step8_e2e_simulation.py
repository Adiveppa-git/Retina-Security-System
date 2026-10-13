import requests

API_BASE = "http://127.0.0.1:8000"

print("=" * 80)
print("TEST 8.E2E: SIMULATE VERIFICATION SUCCESS & ACTION HANDOFF")
print("=" * 80)

# 1. Test CHECK-IN action parameter acceptance on /user/verify_identity
with open("project_images/002.jpeg", "rb") as f:
    resp_checkin = requests.post(
        f"{API_BASE}/user/verify_identity",
        files={"file": f.read()},
        data={"action": "CHECK-IN", "dry_run": "true"}
    )
print("1. Action CHECK-IN Response Status:", resp_checkin.status_code)
assert resp_checkin.status_code == 200

# 2. Test CHECK-OUT action parameter acceptance on /user/verify_identity
with open("project_images/002.jpeg", "rb") as f:
    resp_checkout = requests.post(
        f"{API_BASE}/user/verify_identity",
        files={"file": f.read()},
        data={"action": "CHECK-OUT", "dry_run": "true"}
    )
print("2. Action CHECK-OUT Response Status:", resp_checkout.status_code)
assert resp_checkout.status_code == 200

# 3. Verify Active Sessions Query Endpoint
resp_active = requests.get(f"{API_BASE}/iris/active_sessions")
print("3. Active Sessions Query Status:", resp_active.status_code)
print("   Active Users Count:", resp_active.json().get("total_inside"))
assert resp_active.status_code == 200

print("=" * 80)
print("BACKEND ENDPOINTS CONFIRMED OPERATIONAL FOR STEP 8")
print("=" * 80)
