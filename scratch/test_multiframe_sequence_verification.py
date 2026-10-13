import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbeddingV2
from app.services.iris_recognition_service import recognize_iris_v2_sequence_diagnostic

def test_multiframe_sequential_voting():
    print("==================================================")
    print(" TESTING MULTI-FRAME TEMPORAL VOTING PIPELINE")
    print("==================================================")

    db = SessionLocal()
    try:
        # Test 1: Renkigouda 5-Frame Sequence (Webcam Frame 020.jpeg)
        seq_renki = [
            "project_images/020.jpeg",
            "project_images/020.jpeg",
            "project_images/020.jpeg",
            "project_images/020.jpeg",
            "project_images/020.jpeg"
        ]

        res1 = recognize_iris_v2_sequence_diagnostic(seq_renki)
        print("\n--- Renkigouda 5-Frame Sequence Result ---")
        print("Sequence Status    :", res1["final_diagnostic_status"])
        print("Winner Candidate   :", res1["winning_user_name"], f"(ID #{res1['winning_user_id']})")
        print("Winner Score       :", res1["winning_user_score"])
        print("Candidate Margin   :", res1["candidate_margin"])
        print("Temporal Votes     :", res1["temporal_votes"])
        print("Decision Reason    :", res1["decision_reason"])
        print("Hardware Action    :", res1["hardware_action"])

        # Test 2: Adiveppa 5-Frame Sequence (Webcam Frame 008.jpeg)
        seq_adiveppa = [
            "project_images/008.jpeg",
            "project_images/008.jpeg",
            "project_images/008.jpeg",
            "project_images/008.jpeg",
            "project_images/008.jpeg"
        ]

        res2 = recognize_iris_v2_sequence_diagnostic(seq_adiveppa)
        print("\n--- Adiveppa 5-Frame Sequence Result ---")
        print("Sequence Status    :", res2["final_diagnostic_status"])
        print("Winner Candidate   :", res2["winning_user_name"], f"(ID #{res2['winning_user_id']})")
        print("Winner Score       :", res2["winning_user_score"])
        print("Candidate Margin   :", res2["candidate_margin"])
        print("Temporal Votes     :", res2["temporal_votes"])
        print("Decision Reason    :", res2["decision_reason"])
        print("Hardware Action    :", res2["hardware_action"])

        # Test 3: Unregistered Person 5-Frame Sequence (Frame 016.jpeg)
        seq_unreg = [
            "project_images/016.jpeg",
            "project_images/016.jpeg",
            "project_images/016.jpeg",
            "project_images/016.jpeg",
            "project_images/016.jpeg"
        ]

        res3 = recognize_iris_v2_sequence_diagnostic(seq_unreg)
        print("\n--- Unregistered Person 5-Frame Sequence Result ---")
        print("Sequence Status    :", res3["final_diagnostic_status"])
        print("Winner Candidate   :", res3["winning_user_name"], f"(ID #{res3['winning_user_id']})")
        print("Winner Score       :", res3["winning_user_score"])
        print("Candidate Margin   :", res3["candidate_margin"])
        print("Temporal Votes     :", res3["temporal_votes"])
        print("Decision Reason    :", res3["decision_reason"])
        print("Hardware Action    :", res3["hardware_action"])

    finally:
        db.close()

if __name__ == "__main__":
    test_multiframe_sequential_voting()
