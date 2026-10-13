import requests
import json
import io
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def create_dummy_jpeg():
    img = Image.new("RGB", (320, 240), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_flow():
    print("==================================================")
    print("TESTING CHECK-IN -> INSIDE FLOW & SAFETY RULES")
    print("==================================================")

    jpeg_bytes = create_dummy_jpeg()

    # 1. Test /iris/active_sessions
    res = requests.get(f"{BASE_URL}/iris/active_sessions")
    print(f"1. Active Sessions Status Code: {res.status_code}")
    print(f"   Response: {json.dumps(res.json(), indent=2)}")

    active_data = res.json()
    active_users = active_data.get("active_users", [])

    if active_users:
        first_user = active_users[0]
        user_id = first_user["user_id"]
        user_name = first_user["user_name"]
        print(f"\nFound active user inside: '{user_name}' (#{user_id})")

        # 2. Test DUPLICATE CHECK-IN RULE: Active user tries CHECK-IN again
        print(f"\n2. Testing DUPLICATE CHECK-IN RULE for User #{user_id}...")
        res_dup = requests.post(
            f"{BASE_URL}/iris/verify_1to1",
            data={"claimed_user_id": user_id, "action": "CHECK-IN"},
            files={"file": ("dummy.jpg", jpeg_bytes, "image/jpeg")}
        )
        print(f"   Response Code: {res_dup.status_code}")
        dup_json = res_dup.json()
        print(f"   Decision Reason: {dup_json.get('decision_reason')}")
        print(f"   Final Status: {dup_json.get('final_diagnostic_status')}")

        assert dup_json.get("decision_reason") == "USER_ALREADY_INSIDE", "FAILED: Duplicate check-in was not blocked!"
        assert dup_json.get("final_diagnostic_status") == "CLAIMED_IDENTITY_NOT_VERIFIED", "FAILED: Status should be NOT VERIFIED!"
        print("   SUCCESS: Duplicate check-in request was blocked with USER_ALREADY_INSIDE!")

        # 3. Test 5-Frame Sequence Duplicate Check-In
        print(f"\n3. Testing 5-Frame DUPLICATE CHECK-IN for User #{user_id}...")
        files = [("files", (f"frame_{i}.jpg", jpeg_bytes, "image/jpeg")) for i in range(5)]
        res_seq_dup = requests.post(
            f"{BASE_URL}/iris/verify_sequence_1to1",
            data={"claimed_user_id": user_id, "action": "CHECK-IN"},
            files=files
        )
        print(f"   Response Code: {res_seq_dup.status_code}")
        seq_dup_json = res_seq_dup.json()
        print(f"   Decision Reason: {seq_dup_json.get('decision_reason')}")
        print(f"   Final Status: {seq_dup_json.get('final_diagnostic_status')}")
        assert seq_dup_json.get("decision_reason") == "USER_ALREADY_INSIDE", "FAILED: 5-frame duplicate check-in not blocked!"
        print("   SUCCESS: 5-frame duplicate check-in request was blocked with USER_ALREADY_INSIDE!")

    else:
        print("\nNo active users inside currently. Testing check-out on non-inside user...")
        res_out = requests.post(
            f"{BASE_URL}/iris/verify_1to1",
            data={"claimed_user_id": 71, "action": "CHECK-OUT"},
            files={"file": ("dummy.jpg", jpeg_bytes, "image/jpeg")}
        )
        print(f"   Response Code: {res_out.status_code}")
        out_json = res_out.json()
        print(f"   Decision Reason: {out_json.get('decision_reason')}")
        assert out_json.get("decision_reason") == "USER_NOT_INSIDE", "FAILED: Non-inside check-out was not blocked!"
        print("   SUCCESS: Check-out on non-inside user was blocked with USER_NOT_INSIDE!")

if __name__ == "__main__":
    test_flow()
