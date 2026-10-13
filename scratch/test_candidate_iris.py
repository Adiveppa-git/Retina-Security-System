import sys, os
sys.path.insert(0, 'backend')
sys.path.insert(0, '.')
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

print("--- TESTING CANDIDATE_ONLY IRIS ON PROJECT_IMAGES ---")
for img_name in sorted(os.listdir('project_images')):
    p = os.path.join('project_images', img_name)
    r6 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=6, candidate_only=True, dry_run=True)
    r7 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=7, candidate_only=True, dry_run=True)
    s6 = r6.get('best_similarity', 0.0)
    s7 = r7.get('best_similarity', 0.0)
    b6 = r6.get('biometric_verified', False)
    b7 = r7.get('biometric_verified', False)
    t6 = r6.get('template_ids_used', [])
    t7 = r7.get('template_ids_used', [])
    if s6 > 0.4 or s7 > 0.4:
        print(f"{img_name:<10} | U6: score={s6:.4f}, bio={b6}, temps={t6} | U7: score={s7:.4f}, bio={b7}, temps={t7}")
