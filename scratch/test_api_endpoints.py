import requests

print("==================================================")
print(" TESTING FASTAPI 1:1 API ENDPOINTS")
print("==================================================")

base_url = "http://localhost:8000"

# Test 1: Single Frame 1:1 Endpoint (/iris/verify_1to1)
with open("project_images/020.jpeg", "rb") as f:
    files = {"file": ("test_frame.jpg", f, "image/jpeg")}
    data = {"claimed_user_id": "70"}
    res1 = requests.post(f"{base_url}/iris/verify_1to1", files=files, data=data)

print(f" 1. POST /iris/verify_1to1 Status Code : HTTP {res1.status_code}")
print(f"    Payload Response                    : {res1.json().get('final_diagnostic_status')} | Reason: {res1.json().get('decision_reason')}")

# Test 2: Multi-Frame 1:1 Endpoint (/iris/verify_sequence_1to1)
files_seq = [
    ("files", ("seq_0.jpg", open("project_images/020.jpeg", "rb"), "image/jpeg")),
    ("files", ("seq_1.jpg", open("project_images/020.jpeg", "rb"), "image/jpeg"))
]
data_seq = {"claimed_user_id": "70"}
res2 = requests.post(f"{base_url}/iris/verify_sequence_1to1", files=files_seq, data=data_seq)

print(f" 2. POST /iris/verify_sequence_1to1 Code: HTTP {res2.status_code}")
print(f"    Payload Response                    : {res2.json().get('final_diagnostic_status')} | Reason: {res2.json().get('decision_reason')}")

print("==================================================")
