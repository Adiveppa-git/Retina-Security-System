import cv2
import requests
import time

cap = cv2.VideoCapture(0)
# Warm up
for _ in range(10):
    ret, frame = cap.read()
    time.sleep(0.05)

for attempt in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('live_probe.jpg', img_encoded.tobytes(), 'image/jpeg')}
        resp = requests.post('http://127.0.0.1:8000/user/verify_identity', files=files, data={'dry_run': 'true'})
        data = resp.json()
        f_stage = data.get("face_stage", {})
        i_stage = data.get("iris_stage", {})
        print(f"Attempt {attempt+1}:")
        print(f"  Face Matched: {f_stage.get('matched')}")
        print(f"  Face Score: {f_stage.get('score')} (threshold: {f_stage.get('threshold')})")
        print(f"  Face Candidate: {f_stage.get('candidate_user_name')} (#{f_stage.get('candidate_user_id')})")
        print(f"  Iris Executed: {i_stage.get('executed')}")
        print(f"  Iris Matched: {i_stage.get('matched')}")
        print(f"  Iris Score: {i_stage.get('score')} (threshold: {i_stage.get('threshold')})")
        print(f"  Iris Template IDs: {i_stage.get('template_ids_used')}")
        print(f"  Final Decision: {data.get('final_decision')}")
        print(f"  Failure Stage: {data.get('failure_stage')}")
        print(f"  Hardware Action: {data.get('hardware_action')}")
        if f_stage.get('matched'):
            break
    time.sleep(0.4)

cap.release()
