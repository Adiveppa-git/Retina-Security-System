import sys
import os
import numpy as np
from PIL import Image

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_path = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2, FaceEmbedding, AccessLog
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

def run_template_audit():
    print("=" * 100)
    print("READ-ONLY AUDIT: SUSPICIOUS IRIS TEMPLATES IN DATABASE")
    print("=" * 100)
    
    # 1. Compute reference empty/black crop embedding
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    empty_pil = Image.fromarray(empty_arr)
    black_ref_vec = np.array(extract_v2_embedding(empty_pil), dtype=np.float32)
    black_ref_norm = float(np.linalg.norm(black_ref_vec))
    
    db = SessionLocal()
    try:
        users = {u.id: u.name for u in db.query(User).all()}
        templates = db.query(IrisEmbeddingV2).order_by(IrisEmbeddingV2.id.asc()).all()
        
        print(f"\nTotal Enrolled V2 Templates Audited: {len(templates)}")
        print(f"Total Enrolled Users: {len(users)}\n")
        
        headers = ["User ID", "User Name", "Template ID", "Side", "Dim", "Norm", "Sim to Black", "Status", "Reason Flagged"]
        row_fmt = "{:<8} | {:<12} | {:<11} | {:<4} | {:<4} | {:<6} | {:<12} | {:<8} | {:<30}"
        print(row_fmt.format(*headers))
        print("-" * 115)
        
        flagged_count = 0
        valid_count = 0
        
        for t in templates:
            vec = np.array(t.embedding, dtype=np.float32)
            dim = len(vec)
            norm = float(np.linalg.norm(vec))
            sim_black = float(np.dot(vec, black_ref_vec) / (norm * black_ref_norm)) if norm > 0 else 0.0
            
            reasons = []
            if dim != 512:
                reasons.append(f"INVALID_DIMENSION_{dim}")
            if abs(norm - 1.0) > 0.01:
                reasons.append(f"INVALID_NORM_{norm:.3f}")
            if sim_black >= 0.95:
                reasons.append(f"EMPTY_BLACK_CROP_MATCH ({sim_black:.4f})")
                
            u_name = users.get(t.user_id, f"Unknown ({t.user_id})")
            
            if reasons:
                status = "FLAGGED"
                reason_str = "; ".join(reasons)
                flagged_count += 1
            else:
                status = "VALID"
                reason_str = "OK (Genuine eye crop)"
                valid_count += 1
                
            print(row_fmt.format(
                t.user_id,
                u_name,
                t.id,
                t.eye_side,
                dim,
                f"{norm:.4f}",
                f"{sim_black:.4f}",
                status,
                reason_str
            ))
            
        print("-" * 115)
        print(f"\nAUDIT SUMMARY:")
        print(f"  Valid Templates:      {valid_count} / {len(templates)}")
        print(f"  Flagged Suspicious:   {flagged_count} / {len(templates)}")
        print(f"  Integrity Note:       Read-only audit completed. Database untouched. 0 records deleted/altered.")
        print("=" * 100)
        
    finally:
        db.close()

if __name__ == "__main__":
    run_template_audit()
