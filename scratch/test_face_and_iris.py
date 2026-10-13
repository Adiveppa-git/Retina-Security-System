import cv2
import requests
import time

cap = cv2.VideoCapture(0)
for _ in range(10): cap.read()

for attempt in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        _, img_enc = cv2.imencode('.jpg', frame)
        with open('project_images/002.jpeg', 'rb') as f_iris:
            files = {
                'file': ('live_face.jpg', img_enc.tobytes(), 'image/jpeg'),
                'iris_file': ('probe_iris.jpg', f_iris.read(), 'image/jpeg')
            }
            data = {'action': 'CHECK-IN', 'dry_run': 'true'}
            resp = requests.post('http://127.0.0.1:8000/user/verify_identity', files=files, data=data)
            res = resp.json()
            f = res.get("face_stage", {})
            print(f"Attempt {attempt + 1}: num_faces={f.get('num_faces')}, matched={f.get('matched')}, candidate={f.get('candidate_user_name')}, score={f.get('score')}")
    time.sleep(0.3)

cap.release()
