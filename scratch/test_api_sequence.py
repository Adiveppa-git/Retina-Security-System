import urllib.request
import json
import os

url = 'http://127.0.0.1:8000/iris/verify_sequence_1to1'
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'

image_files = [
    'project_images/008.jpeg',
    'project_images/008.jpeg',
    'project_images/008.jpeg',
    'project_images/008.jpeg',
    'project_images/008.jpeg'
]

body = bytearray()

print("Capturing 5 distinct frame files for API verification test:")
for idx, img_path in enumerate(image_files):
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue
    with open(img_path, 'rb') as f:
        content = f.read()
    print(f"Frame {idx + 1} captured | size = {len(content)} bytes ({os.path.basename(img_path)})")
    
    body.extend(f'--{boundary}\r\n'.encode('utf-8'))
    body.extend(f'Content-Disposition: form-data; name="files"; filename="seq_frame_{idx}.jpg"\r\n'.encode('utf-8'))
    body.extend(b'Content-Type: image/jpeg\r\n\r\n')
    body.extend(content)
    body.extend(b'\r\n')

body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend(b'Content-Disposition: form-data; name="claimed_user_id"\r\n\r\n')
body.extend(b'71\r\n')
body.extend(f'--{boundary}--\r\n'.encode('utf-8'))

print("\nSending 5 unique frames to /iris/verify_sequence_1to1...")

req = urllib.request.Request(url, data=bytes(body))
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    response = urllib.request.urlopen(req)
    res_data = json.loads(response.read().decode('utf-8'))
    print("\nHTTP Status Code: 200 OK")
    print("API Response Output:")
    print(f"  - total_frames_processed : {res_data.get('total_frames_processed')}")
    print(f"  - valid_frames_count     : {res_data.get('valid_frames_count')}")
    print(f"  - passing_frames_count   : {res_data.get('passing_frames_count')}")
    print(f"  - failing_frames_count   : {res_data.get('failing_frames_count')}")
    print(f"  - final_diagnostic_status: {res_data.get('final_diagnostic_status')}")
    print(f"  - decision_reason        : {res_data.get('decision_reason')}")
    print(f"  - best_similarity        : {res_data.get('best_similarity')}")
    print(f"  - strongest_competing    : {res_data.get('strongest_competing_score')}")
    print(f"  - identity_margin        : {res_data.get('identity_margin')}")
    print(f"  - hardware_action        : {res_data.get('hardware_action')}")
except Exception as e:
    print("Request error:", e)
