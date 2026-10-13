import requests
import io
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

print("--- TESTING POST /iris/verify_sequence_1to1 ENDPOINT ---")

# 1. Fetch user ID
users = requests.get(f"{BASE_URL}/user/all").json()
if not users:
    print("No users found.")
    exit(1)

claimed_id = users[0]["id"]
print(f"Testing sequence verification with Claimed User ID: {claimed_id} ({users[0]['name']})...")

# 2. Create 5 sample image files in-memory
files_payload = []
for i in range(5):
    img = Image.new("RGB", (320, 240), color=(10 * i, 100 + i * 20, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files_payload.append(("files", (f"seq_frame_{i}.jpg", buf.getvalue(), "image/jpeg")))

# 3. Send POST request
data = {
    "claimed_user_id": str(claimed_id),
    "action": "CHECK-IN"
}

response = requests.post(f"{BASE_URL}/iris/verify_sequence_1to1", files=files_payload, data=data)

print("HTTP Status Code:", response.status_code)
if response.status_code == 200:
    res_json = response.json()
    print("Response JSON:")
    print("  total_frames_processed:", res_json.get("total_frames_processed"))
    print("  valid_frames_count:", res_json.get("valid_frames_count"))
    print("  passing_frames_count:", res_json.get("passing_frames_count"))
    print("  failing_frames_count:", res_json.get("failing_frames_count"))
    print("  best_similarity:", res_json.get("best_similarity"))
    print("  final_diagnostic_status:", res_json.get("final_diagnostic_status"))
    print("  decision_reason:", res_json.get("decision_reason"))
    print("\nSUCCESS: Backend processed 5 sequence frames cleanly!")
else:
    print("ERROR RESPONSE:", response.text)
