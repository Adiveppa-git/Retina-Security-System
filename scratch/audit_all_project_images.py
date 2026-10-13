import sys, os
sys.path.insert(0, 'backend')
sys.path.insert(0, '.')
from app.services.face_recognition_service import identify_user_by_face
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

print("=" * 100)
print(f"{'Image':<12} | {'Face Match':<10} | {'Cand ID':<8} | {'Face Name':<12} | {'Face Sc':<8} | {'U6 Iris Sc':<10} | {'U7 Iris Sc':<10}")
print("=" * 100)

for f in sorted(os.listdir('project_images')):
    if not f.endswith(('.jpeg', '.jpg', '.png')): continue
    p = os.path.join('project_images', f)
    f_res = identify_user_by_face(p)
    ir6 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=6, candidate_only=True, dry_run=True)
    ir7 = recognize_iris_v2_1to1_diagnostic(p, claimed_user_id=7, candidate_only=True, dry_run=True)
    
    f_matched = f_res.get('matched', False)
    f_uid = str(f_res.get('candidate_user_id'))
    f_name = str(f_res.get('candidate_user_name'))
    f_sc = f_res.get('face_score', 0.0)
    ir6_sc = ir6.get('best_similarity', 0.0)
    ir7_sc = ir7.get('best_similarity', 0.0)
    
    print(f"{f:<12} | {str(f_matched):<10} | {f_uid:<8} | {f_name:<12} | {f_sc:<8.4f} | {ir6_sc:<10.4f} | {ir7_sc:<10.4f}")
