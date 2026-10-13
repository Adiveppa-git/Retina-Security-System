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

def run_1to1_verification_test_suite():
    print("==================================================")
    print(" EXECUTING ANTI-FALSE-ACCEPT COMPETITOR GUARD SUITE")
    print("==================================================")

    db = SessionLocal()
    try:
        u70 = db.query(User).filter(User.id == 70).first()
        u71 = db.query(User).filter(User.id == 71).first()

        print(f"Enrolled DB Users Audit: User #70='{u70.name if u70 else 'N/A'}' | User #71='{u71.name if u71 else 'N/A'}'\n")

        test_cases = [
            ("TEST A: Claimed Renkigouda (#70), Present Renkigouda", 70, "project_images/020.jpeg"),
            ("TEST B: Claimed Adiveppa (#71), Present Adiveppa", 71, "project_images/008.jpeg"),
            ("TEST C: Claimed Adiveppa (#71), Present Renkigouda", 71, "project_images/020.jpeg"),
            ("TEST D: Claimed Renkigouda (#70), Present Adiveppa", 70, "project_images/008.jpeg"),
            ("TEST E: Claimed Adiveppa (#71), Present Unregistered", 71, "project_images/016.jpeg"),
            ("TEST F: Claimed Renkigouda (#70), Present Unregistered", 70, "project_images/016.jpeg")
        ]

        print("-" * 160)
        print(f"| Test Description                                        | Claimed User | Input Image   | Claimed Score | Competitor | Margin  | Status Code                   | Decision Reason               | HW Action |")
        print("-" * 160)

        for label, claimed_id, img_path in test_cases:
            if not os.path.exists(img_path):
                print(f"Skipping {label} (File not found)")
                continue

            res = recognize_iris_v2_1to1_diagnostic(img_path, claimed_id)

            actual_status = res["final_diagnostic_status"]
            reason = res.get("decision_reason", "UNKNOWN")
            c_score = res.get("best_similarity", 0.0)
            comp_score = res.get("strongest_competing_score", 0.0)
            margin = res.get("identity_margin", 0.0)
            hw = res.get("hardware_action", "UNKNOWN")

            claimed_name = u70.name if claimed_id == 70 else (u71.name if claimed_id == 71 else "Unknown")
            print(f"| {label:<56} | {claimed_name:<12} | {os.path.basename(img_path):<13} | {c_score:.4f}        | {comp_score:.4f}     | {margin:+.4f} | {actual_status:<29} | {reason:<29} | {hw:<9} |")

        print("-" * 160)

        # TEST G: 5-Frame Sequence Anti-False-Accept Test with Wrong Person
        print("\n--- TEST G: 5-Frame Sequence Anti-False-Accept Test ---")
        seq_frames = ["project_images/016.jpeg"] * 5
        res_seq = recognize_iris_v2_sequence_1to1_diagnostic(seq_frames, 71)
        print(f" Claimed User           : {res_seq['claimed_user_name']} (#71)")
        print(f" Sequence Status        : {res_seq['final_diagnostic_status']}")
        print(f" Decision Reason        : {res_seq['decision_reason']}")
        print(f" Aggregated Score       : {res_seq['best_similarity']}")
        print(f" Strongest Competitor   : {res_seq['strongest_competing_score']}")
        print(f" Identity Margin        : {res_seq['identity_margin']}")
        print(f" Passing Frames Count   : {res_seq['passing_frames_count']} / {res_seq['valid_frames_count']}")
        print(f" Hardware Control Status: {res_seq['hardware_action']}")

        print("\n==================================================")
        print(" ANTI-FALSE-ACCEPT COMPETITOR GUARD SUITE COMPLETE")
        print("==================================================\n")

    finally:
        db.close()

if __name__ == "__main__":
    run_1to1_verification_test_suite()
