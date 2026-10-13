import sys
import os

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic
from PIL import Image

test_imgs = [
    'project_images/003.jpeg',
    'project_images/004.jpeg',
    'project_images/005.jpeg',
    'project_images/014.jpeg',
    'project_images/015.jpeg',
    'scratch/live_sample_1.jpg',
    'scratch/live_sample_3.jpg',
    'scratch/live_sample_4.jpg',
    'scratch/live_sample_8.jpg'
]

for p in test_imgs:
    if os.path.exists(p):
        img = Image.open(p)
        r6 = recognize_iris_v2_1to1_diagnostic(img, claimed_user_id=6, action='DIAGNOSTIC')
        r7 = recognize_iris_v2_1to1_diagnostic(img, claimed_user_id=7, action='DIAGNOSTIC')
        print(f"{p}:")
        print(f"   User 6: score={r6['best_similarity']:.4f}, verified={r6['biometric_verified']}, status={r6['final_diagnostic_status']}")
        print(f"   User 7: score={r7['best_similarity']:.4f}, verified={r7['biometric_verified']}, status={r7['final_diagnostic_status']}")
