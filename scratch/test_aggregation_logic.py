import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2
from app.services.iris_detection_service import detect_and_crop_iris
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

def test_aggregation():
    db = SessionLocal()
    try:
        print("==================================================")
        print(" TESTING USER-LEVEL MULTI-TEMPLATE AGGREGATION ENGINE")
        print("==================================================")

        test_frames = [
            ("Renkigouda Test Frame 002.jpeg", "project_images/002.jpeg"),
            ("Renkigouda Test Frame 020.jpeg", "project_images/020.jpeg"),
        ]

        device = torch.device("cpu")

        for label, img_path in test_frames:
            print(f"\n--- {label} ({img_path}) ---")
            pil_img = Image.open(img_path).convert("RGB")
            res = detect_and_crop_iris(np.array(pil_img))
            payload = res.payload

            lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
            rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]

            left_crop = Image.fromarray(np.array(pil_img)[ly1:ly2, lx1:lx2])
            right_crop = Image.fromarray(np.array(pil_img)[ry1:ry2, rx1:rx2])

            left_vec = extract_v2_embedding(left_crop)
            right_vec = extract_v2_embedding(right_crop)

            left_t = torch.tensor(left_vec, device=device).unsqueeze(0)
            right_t = torch.tensor(right_vec, device=device).unsqueeze(0)

            users = db.query(User).all()
            user_scores = []

            for u in users:
                l_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == u.id, IrisEmbeddingV2.eye_side == "L").all()
                r_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == u.id, IrisEmbeddingV2.eye_side == "R").all()

                l_sims = [float(torch.nn.functional.cosine_similarity(left_t, torch.tensor(r.embedding, device=device).unsqueeze(0), dim=1).item()) for r in l_recs]
                r_sims = [float(torch.nn.functional.cosine_similarity(right_t, torch.tensor(r.embedding, device=device).unsqueeze(0), dim=1).item()) for r in r_recs]

                l_top2 = sorted(l_sims, reverse=True)[:min(2, len(l_sims))] if l_sims else [0.0]
                r_top2 = sorted(r_sims, reverse=True)[:min(2, len(r_sims))] if r_sims else [0.0]

                l_agg = float(np.mean(l_top2)) if l_sims else 0.0
                r_agg = float(np.mean(r_top2)) if r_sims else 0.0

                l_best = max(l_sims) if l_sims else 0.0
                r_best = max(r_sims) if r_sims else 0.0

                combined = (l_agg + r_agg) / 2.0 if (l_sims and r_sims) else (l_agg or r_agg)

                user_scores.append({
                    "user_id": u.id,
                    "user_name": u.name,
                    "left_count": len(l_recs),
                    "right_count": len(r_recs),
                    "l_best": l_best,
                    "l_agg": l_agg,
                    "r_best": r_best,
                    "r_agg": r_agg,
                    "combined": combined
                })

            user_scores.sort(key=lambda x: x["combined"], reverse=True)
            winner = user_scores[0]
            second = user_scores[1] if len(user_scores) > 1 else None
            margin = winner["combined"] - (second["combined"] if second else 0.0)

            print("User Scores Ranking:")
            for cand in user_scores:
                print(f"  Candidate User #{cand['user_id']} ({cand['user_name']:<12}): L_best={cand['l_best']:.4f}, L_agg={cand['l_agg']:.4f} | R_best={cand['r_best']:.4f}, R_agg={cand['r_agg']:.4f} | Combined Score = {cand['combined']:.4f}")
            
            print(f"  WINNER: {winner['user_name']} (ID #{winner['user_id']}) with Combined Score = {winner['combined']:.4f} | Margin over 2nd place = {margin:.4f}")

    finally:
        db.close()

if __name__ == "__main__":
    test_aggregation()
