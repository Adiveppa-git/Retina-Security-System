import sys, os, cv2
sys.path.insert(0, 'backend')
sys.path.insert(0, '.')
from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2
from app.services.face_recognition_service import identify_user_by_face
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

db = SessionLocal()
print("--- FACE IDENTIFY RESULTS ---")
for img_name in ['002.jpeg', '008.jpeg', '014.jpeg', '015.jpeg', '020.jpeg']:
    p = os.path.join('project_images', img_name)
    res = identify_user_by_face(p, db=db)
    print(f"Face: {img_name} -> matched={res['matched']}, cand_id={res.get('candidate_user_id')}, name={res.get('candidate_user_name')}, score={res.get('face_score')}")

print("\n--- IRIS 1:1 RECOGNITION (READ ONLY DRY-RUN) ---")
# User 6 iris templates:
for img_name in ['002.jpeg', '008.jpeg', '020.jpeg']:
    p = os.path.join('project_images', img_name)
    res6 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=6, action="DIAGNOSTIC")
    res7 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=7, action="DIAGNOSTIC")
    print(f"Iris: {img_name} for User 6: score={res6.get('best_similarity')}, status={res6.get('final_diagnostic_status')}, bio={res6.get('biometric_verified')}")
    print(f"Iris: {img_name} for User 7: score={res7.get('best_similarity')}, status={res7.get('final_diagnostic_status')}, bio={res7.get('biometric_verified')}")
