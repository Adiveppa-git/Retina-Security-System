import sys, os
sys.path.insert(0, 'backend')
sys.path.insert(0, '.')
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

for uid in [6, 7]:
    print(f'=== Testing Iris for User #{uid} ===')
    for img_name in sorted(os.listdir('project_images')):
        if img_name.endswith(('.jpeg', '.jpg', '.png')):
            p = os.path.join('project_images', img_name)
            try:
                res = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=uid)
                sc = res.get('best_similarity', 0.0)
                status = res.get('final_diagnostic_status')
                bio = res.get('biometric_verified')
                print(f"User {uid} | {img_name}: score={sc:.4f}, status={status}, biometrics_passed={bio}")
            except Exception as e:
                print(f"User {uid} | {img_name}: err={e}")
