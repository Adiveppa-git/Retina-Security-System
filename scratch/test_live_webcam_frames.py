import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2

def test_live_webcam_templates_cross_matching():
    db = SessionLocal()
    try:
        print("==================================================")
        print(" LIVE WEBCAM ENROLLED TEMPLATE CROSS-VERIFICATION ")
        print("==================================================")

        u70_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 70).all()
        u71_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 71).all()

        device = torch.device("cpu")

        # Function to evaluate a query vector against all enrolled users using User-Level Aggregation
        def evaluate_vector(query_vec, query_eye, source_label, true_user_id):
            query_t = torch.tensor(query_vec, device=device).unsqueeze(0)
            users = db.query(User).all()
            user_scores = []

            for u in users:
                target_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == u.id, IrisEmbeddingV2.eye_side == query_eye).all()
                if not target_recs:
                    continue

                sims = [float(torch.nn.functional.cosine_similarity(query_t, torch.tensor(r.embedding, device=device).unsqueeze(0), dim=1).item()) for r in target_recs]
                top2 = sorted(sims, reverse=True)[:min(2, len(sims))]
                top2_mean = float(np.mean(top2))
                best_sim = max(sims)

                user_scores.append({
                    "user_id": u.id,
                    "user_name": u.name,
                    "best_sim": best_sim,
                    "agg_sim": top2_mean
                })

            user_scores.sort(key=lambda x: x["agg_sim"], reverse=True)
            winner = user_scores[0]
            second = user_scores[1] if len(user_scores) > 1 else None
            margin = winner["agg_sim"] - (second["agg_sim"] if second else 0.0)

            is_correct = (winner["user_id"] == true_user_id)
            status_str = "CORRECT MATCH" if is_correct else "FALSE MATCH"

            print(f" {source_label:<32} (Eye: {query_eye}): Winner = {winner['user_name']:<12} (ID #{winner['user_id']}) | Score = {winner['agg_sim']:.4f} | 2nd Place = {second['user_name'] if second else 'None'} ({second['agg_sim'] if second else 0.0:.4f}) | Margin = {margin:.4f} | Status = {status_str}")

        print("\nEvaluating Renkigouda Enrolled Live Webcam Templates against Database:")
        for rec in u70_recs:
            evaluate_vector(rec.embedding, rec.eye_side, f"Renkigouda Template #{rec.id}", 70)

        print("\nEvaluating Adiveppa Enrolled Live Webcam Templates against Database:")
        for rec in u71_recs:
            evaluate_vector(rec.embedding, rec.eye_side, f"Adiveppa Template #{rec.id}", 71)

    finally:
        db.close()

if __name__ == "__main__":
    test_live_webcam_templates_cross_matching()
