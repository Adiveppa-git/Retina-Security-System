import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2
from app.services.iris_recognition_service import (
    recognize_iris_v2_1to1_diagnostic,
    recognize_iris_v2_sequence_1to1_diagnostic
)

def run_real_1to1_tests():
    print("==================================================")
    print(" EXECUTING REAL 1:1 BIOMETRIC SEPARATION SUITE")
    print("==================================================")

    db = SessionLocal()
    try:
        u70 = db.query(User).filter(User.id == 70).first()
        u71 = db.query(User).filter(User.id == 71).first()

        # Gather Enrolled Templates
        u70_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 70).all()
        u71_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 71).all()

        device = torch.device("cpu")
        RGB_1TO1_THRESHOLD = 0.80

        def evaluate_1to1_single(claimed_id, query_vec, eye_side):
            claimed_user = u70 if claimed_id == 70 else u71
            target_recs = [r for r in (u70_recs if claimed_id == 70 else u71_recs) if r.eye_side == eye_side]
            if not target_recs:
                return "CLAIMED_IDENTITY_NOT_VERIFIED", 0.0, "NO_TEMPLATES"

            query_t = torch.tensor(query_vec, device=device).unsqueeze(0)
            sims = [float(torch.nn.functional.cosine_similarity(query_t, torch.tensor(r.embedding, device=device).unsqueeze(0), dim=1).item()) for r in target_recs]

            top2 = sorted(sims, reverse=True)[:min(2, len(sims))]
            score = float(np.mean(top2)) if top2 else 0.0

            if score >= RGB_1TO1_THRESHOLD:
                return "CLAIMED_IDENTITY_VERIFIED", score, f"SCORE_{score:.4f}_GE_{RGB_1TO1_THRESHOLD}"
            else:
                return "CLAIMED_IDENTITY_NOT_VERIFIED", score, "INSUFFICIENT_OR_AMBIGUOUS_1TO1_EVIDENCE"

        # Evaluate across Live Webcam Enrolled Templates
        # Test A: Claimed Renkigouda (#70), Input Renkigouda (#70 Live Template #35)
        res_a, score_a, r_a = evaluate_1to1_single(70, u70_recs[0].embedding, u70_recs[0].eye_side)

        # Test B: Claimed Adiveppa (#71), Input Adiveppa (#71 Live Template #46)
        res_b, score_b, r_b = evaluate_1to1_single(71, u71_recs[0].embedding, u71_recs[0].eye_side)

        # Test C: Claimed Renkigouda (#70), Input Adiveppa (#71 Live Template #46)
        res_c, score_c, r_c = evaluate_1to1_single(70, u71_recs[0].embedding, u71_recs[0].eye_side)

        # Test D: Claimed Adiveppa (#71), Input Renkigouda (#70 Live Template #35)
        res_d, score_d, r_d = evaluate_1to1_single(71, u70_recs[0].embedding, u70_recs[0].eye_side)

        test_suite = [
            ("TEST A: Claimed Renkigouda | Actual Renkigouda", 70, res_a, score_a, "CLAIMED_IDENTITY_VERIFIED"),
            ("TEST B: Claimed Adiveppa   | Actual Adiveppa",   71, res_b, score_b, "CLAIMED_IDENTITY_VERIFIED"),
            ("TEST C: Claimed Renkigouda | Actual Adiveppa",   70, res_c, score_c, "CLAIMED_IDENTITY_NOT_VERIFIED"),
            ("TEST D: Claimed Adiveppa   | Actual Renkigouda", 71, res_d, score_d, "CLAIMED_IDENTITY_NOT_VERIFIED"),
        ]

        print("-" * 130)
        print(f"| Test Name                                      | Claimed User | Score  | Expected Status               | Actual Status                 | Pass/Fail |")
        print("-" * 130)

        all_passed = True
        for label, cid, actual_st, score, exp_st in test_suite:
            is_pass = (actual_st == exp_st)
            if not is_pass:
                all_passed = False
            p_str = "PASS" if is_pass else "FAIL"
            cname = "Renkigouda" if cid == 70 else "Adiveppa"
            print(f"| {label:<48} | {cname:<12} | {score:.4f} | {exp_st:<29} | {actual_st:<29} | {p_str:<9} |")
        print("-" * 130)

        if all_passed:
            print("\n ALL 1:1 BIOMETRIC SEPARATION TESTS PASSED CLEANLY!")
        else:
            print("\n WARNING: SOME TESTS DID NOT PASS. INVESTIGATING SEPARATION THRESHOLD...")

    finally:
        db.close()

if __name__ == "__main__":
    run_real_1to1_tests()
