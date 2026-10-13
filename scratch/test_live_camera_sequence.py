import cv2
import requests
import time

print("=" * 80)
print("LIVE ZEBRONICS RGB WEBCAM TWO-STAGE IDENTITY VERIFICATION TEST")
print("=" * 80)

cap = cv2.VideoCapture(0)
# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for _ in range(15):
    cap.read()
    time.sleep(0.04)

frames = []
for i in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        frames.append(frame)
    time.sleep(0.2)

cap.release()
print(f"Captured {len(frames)} frames from live camera.")

for idx, frame in enumerate(frames):
    _, img_enc = cv2.imencode(".jpg", frame)
    files = {"file": (f"live_frame_{idx}.jpg", img_enc.tobytes(), "image/jpeg")}
    data = {"dry_run": "true", "action": "CHECK-IN"}
    resp = requests.post("http://127.0.0.1:8000/user/verify_identity", files=files, data=data)
    res = resp.json()
    f = res.get("face_stage", {})
    i = res.get("iris_stage", {})
    print(f"\n[Frame {idx + 1}/5]")
    print(f"  Face: Matched={f.get('matched')}, Candidate={f.get('candidate_user_name')} (#{f.get('candidate_user_id')}), Score={f.get('score')}")
    print(f"  Iris: Executed={i.get('executed')}, Score={i.get('score')}, Templates={i.get('template_ids_used')}, Matched={i.get('matched')}")
    print(f"  Final Decision: {res.get('final_decision')}")
    print(f"  Failure Stage: {res.get('failure_stage')}")
    print(f"  Hardware Action: {res.get('hardware_action')}")
