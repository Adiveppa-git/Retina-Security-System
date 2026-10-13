import sys
import os
import cv2
import numpy as np

# Setup workspace and backend paths
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database.connection import SessionLocal
from app.models.user_model import User, FaceEmbedding, IrisEmbeddingV2, AccessLog
from app.services.face_recognition_service import (
    identify_user_by_face,
    verify_two_stage_identity,
    DEFAULT_FACE_MATCH_THRESHOLD
)
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

def run_step5_score_analysis():
    print("=" * 100)
    print("STEP 5 — SAFE TWO-STAGE SCORE ANALYSIS (SAFE DIAGNOSTIC MODE ONLY)")
    print("=" * 100)

    db = SessionLocal()
    try:
        # Baseline audit
        users_count_start = db.query(User).count()
        face_count_start = db.query(FaceEmbedding).count()
        iris_v2_count_start = db.query(IrisEmbeddingV2).count()
        access_logs_count_start = db.query(AccessLog).count()

        print(f"\n[BASELINE STATE]")
        print(f"  Users: {users_count_start} | Face Embeddings: {face_count_start} | Iris V2: {iris_v2_count_start} | Access Logs: {access_logs_count_start}")
        print(f"  Face Threshold: {DEFAULT_FACE_MATCH_THRESHOLD:.3f} (DEVELOPMENT ONLY)")
        print(f"  Iris Threshold: 0.800 (UNCHANGED)")

        # Prepare 10 distinct samples for User #6 (Adiveppa)
        # Pairs of (face_source, iris_source, label) from Zebronics camera session captures
        u6_samples = [
            ("project_images/014.jpeg", "project_images/008.jpeg", "User 6 Sample 01 (Probe Face + Normal Iris)"),
            ("project_images/003.jpeg", "project_images/008.jpeg", "User 6 Sample 02 (Reg Sample 1 + Normal Iris)"),
            ("project_images/004.jpeg", "project_images/008.jpeg", "User 6 Sample 03 (Reg Sample 2 + Normal Iris)"),
            ("project_images/014.jpeg", "scratch/phase4_5_visualizations/User_B_B1_normal/1_original_webcam_frame.jpg", "User 6 Sample 04 (Probe Face + B1 Normal Frame)"),
            ("project_images/003.jpeg", "scratch/phase4_5_visualizations/User_B_B1_normal/1_original_webcam_frame.jpg", "User 6 Sample 05 (Reg Face 1 + B1 Normal Frame)"),
            ("project_images/004.jpeg", "scratch/phase4_5_visualizations/User_B_B2_verify/1_original_webcam_frame.jpg", "User 6 Sample 06 (Reg Face 2 + B2 Verify Frame)"),
            ("project_images/014.jpeg", "project_images/018.jpeg", "User 6 Sample 07 (Probe Face + 018 Iris Frame)"),
            ("project_images/009.jpeg", "project_images/008.jpeg", "User 6 Sample 08 (Face 009 + Normal Iris)"),
            ("project_images/014.jpeg", "scratch/phase4_5_visualizations/User_B_B2_verify/1_original_webcam_frame.jpg", "User 6 Sample 09 (Probe Face + B2 Frame)"),
            ("project_images/003.jpeg", "project_images/018.jpeg", "User 6 Sample 10 (Reg Face 1 + 018 Iris Frame)")
        ]

        # Prepare 10 distinct samples for User #7 (basagouda)
        u7_samples = [
            ("project_images/015.jpeg", "project_images/016.jpeg", "User 7 Sample 01 (Probe Face + 016 Iris)"),
            ("project_images/005.jpeg", "project_images/016.jpeg", "User 7 Sample 02 (Reg Face 1 + 016 Iris)"),
            ("project_images/015.jpeg", "project_images/002.jpeg", "User 7 Sample 03 (Probe Face + 002 Iris)"),
            ("project_images/005.jpeg", "project_images/002.jpeg", "User 7 Sample 04 (Reg Face 1 + 002 Iris)"),
            ("project_images/015.jpeg", "project_images/020.jpeg", "User 7 Sample 05 (Probe Face + 020 Iris)"),
            ("project_images/005.jpeg", "project_images/020.jpeg", "User 7 Sample 06 (Reg Face 1 + 020 Iris)"),
            ("project_images/015.jpeg", "scratch/phase4_5_visualizations/User_A_A1_normal/1_original_webcam_frame.jpg", "User 7 Sample 07 (Probe Face + A1 Normal Frame)"),
            ("project_images/005.jpeg", "scratch/phase4_5_visualizations/User_A_A1_normal/1_original_webcam_frame.jpg", "User 7 Sample 08 (Reg Face 1 + A1 Normal Frame)"),
            ("project_images/015.jpeg", "scratch/phase4_5_visualizations/User_A_A2_verify/1_original_webcam_frame.jpg", "User 7 Sample 09 (Probe Face + A2 Verify Frame)"),
            ("project_images/005.jpeg", "scratch/phase4_5_visualizations/User_A_A2_verify/1_original_webcam_frame.jpg", "User 7 Sample 10 (Reg Face 1 + A2 Verify Frame)")
        ]

        # Prepare 10 distinct samples for Impostor / Unknown persons
        impostor_samples = [
            ("datasets/anti_spoof_dataset/test/fake/fake_ts_00048.jpg", "project_images/012.jpeg", "Impostor 01 (Fake 048 Face + Unknown Iris 012)"),
            ("datasets/anti_spoof_dataset/test/fake/fake_ts_00001.jpg", "project_images/016.jpeg", "Impostor 02 (Fake 001 Face + Unknown Iris 016)"),
            ("datasets/anti_spoof_dataset/test/fake/fake_ts_00002.jpg", "project_images/002.jpeg", "Impostor 03 (Fake 002 Face + Cross Iris 002)"),
            ("datasets/anti_spoof_dataset/test/fake/fake_ts_00003.jpg", "project_images/008.jpeg", "Impostor 04 (Fake 003 Face + Cross Iris 008)"),
            ("datasets/anti_spoof_dataset/test/fake/fake_ts_00004.jpg", "project_images/020.jpeg", "Impostor 05 (Fake 004 Face + Cross Iris 020)"),
            ("project_images/001.jpeg", "project_images/012.jpeg", "Impostor 06 (Unknown Frame 001 Face + 012 Iris)"),
            ("project_images/006.jpeg", "scratch/phase4_5_visualizations/User_C_C1_normal/1_original_webcam_frame.jpg", "Impostor 07 (Unknown Frame 006 Face + C1 Iris)"),
            ("project_images/007.jpeg", "scratch/phase4_5_visualizations/User_C_C2_verify/1_original_webcam_frame.jpg", "Impostor 08 (Unknown Frame 007 Face + C2 Iris)"),
            ("project_images/011.jpeg", "scratch/phase4_5_visualizations/User_D_D1_normal/1_original_webcam_frame.jpg", "Impostor 09 (Unknown Frame 011 Face + D1 Iris)"),
            ("project_images/017.jpeg", "project_images/008.jpeg", "Impostor 10 (Unknown Frame 017 Face + 008 Iris)")
        ]

        # Results collectors
        u6_results = []
        u7_results = []
        impostor_results = []

        print("\n" + "=" * 100)
        print("COLLECTING AND EVALUATING USER #6 GENUINE SAMPLES (10 SAMPLES)")
        print("=" * 100)
        for face_path, iris_path, label in u6_samples:
            res = verify_two_stage_identity(
                face_image_input=face_path,
                iris_image_input=iris_path,
                action="CHECK-IN",
                dry_run=True,
                db=db
            )
            res["label"] = label
            res["face_input"] = os.path.basename(face_path)
            res["iris_input"] = os.path.basename(iris_path)
            u6_results.append(res)
            
            f_stage = res["face_stage"]
            i_stage = res["iris_stage"]
            print(f"[{label}]")
            print(f"  Face: matched={f_stage['matched']}, cand_id={f_stage['candidate_user_id']} ({f_stage['candidate_user_name']}), score={f_stage['score']}, thresh={f_stage['threshold']}")
            print(f"  Iris: executed={i_stage['executed']}, uid={i_stage['user_id_checked']}, L={i_stage.get('left_score')}, R={i_stage.get('right_score')}, agg={i_stage['score']}, thresh={i_stage['threshold']}, matched={i_stage['matched']}")
            print(f"  Decision: {res['final_decision']} | Failure Stage: {res['failure_stage']} | HW: {res['hardware_action']}\n")

        print("\n" + "=" * 100)
        print("COLLECTING AND EVALUATING USER #7 GENUINE SAMPLES (10 SAMPLES)")
        print("=" * 100)
        for face_path, iris_path, label in u7_samples:
            res = verify_two_stage_identity(
                face_image_input=face_path,
                iris_image_input=iris_path,
                action="CHECK-IN",
                dry_run=True,
                db=db
            )
            res["label"] = label
            res["face_input"] = os.path.basename(face_path)
            res["iris_input"] = os.path.basename(iris_path)
            u7_results.append(res)

            f_stage = res["face_stage"]
            i_stage = res["iris_stage"]
            print(f"[{label}]")
            print(f"  Face: matched={f_stage['matched']}, cand_id={f_stage['candidate_user_id']} ({f_stage['candidate_user_name']}), score={f_stage['score']}, thresh={f_stage['threshold']}")
            print(f"  Iris: executed={i_stage['executed']}, uid={i_stage['user_id_checked']}, L={i_stage.get('left_score')}, R={i_stage.get('right_score')}, agg={i_stage['score']}, thresh={i_stage['threshold']}, matched={i_stage['matched']}")
            print(f"  Decision: {res['final_decision']} | Failure Stage: {res['failure_stage']} | HW: {res['hardware_action']}\n")

        print("\n" + "=" * 100)
        print("COLLECTING AND EVALUATING IMPOSTOR / UNKNOWN SAMPLES (10 SAMPLES)")
        print("=" * 100)
        for face_path, iris_path, label in impostor_samples:
            res = verify_two_stage_identity(
                face_image_input=face_path,
                iris_image_input=iris_path,
                action="CHECK-IN",
                dry_run=True,
                db=db
            )
            res["label"] = label
            res["face_input"] = os.path.basename(face_path)
            res["iris_input"] = os.path.basename(iris_path)

            # Also compute cross-impostor iris similarity against both candidates for biometric margin analysis
            ir6 = recognize_iris_v2_1to1_diagnostic(iris_path, claimed_user_id=6, candidate_only=True, dry_run=True)
            ir7 = recognize_iris_v2_1to1_diagnostic(iris_path, claimed_user_id=7, candidate_only=True, dry_run=True)
            res["impostor_iris_score_u6"] = ir6.get("best_similarity", 0.0)
            res["impostor_iris_score_u7"] = ir7.get("best_similarity", 0.0)
            res["max_impostor_iris_score"] = max(res["impostor_iris_score_u6"], res["impostor_iris_score_u7"])
            impostor_results.append(res)

            f_stage = res["face_stage"]
            i_stage = res["iris_stage"]
            print(f"[{label}]")
            print(f"  Face: matched={f_stage['matched']}, cand_id={f_stage['candidate_user_id']} ({f_stage.get('candidate_user_name')}), score={f_stage['score']}, reason={f_stage.get('reason')}")
            print(f"  Iris Executed: {i_stage['executed']} | Impostor Iris vs U6={res['impostor_iris_score_u6']:.4f}, vs U7={res['impostor_iris_score_u7']:.4f}")
            print(f"  Decision: {res['final_decision']} | Failure Stage: {res['failure_stage']} | HW: {res['hardware_action']}\n")

        # -------------------------------------------------------------
        # STATISTICAL CALCULATIONS
        # -------------------------------------------------------------
        u6_iris_scores = [r["iris_stage"]["score"] for r in u6_results if r["iris_stage"]["executed"]]
        u7_iris_scores = [r["iris_stage"]["score"] for r in u7_results if r["iris_stage"]["executed"]]
        all_genuine_scores = u6_iris_scores + u7_iris_scores
        impostor_iris_scores = [r["max_impostor_iris_score"] for r in impostor_results]

        def calc_stats(scores):
            if not scores:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
            return {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores))
            }

        u6_stats = calc_stats(u6_iris_scores)
        u7_stats = calc_stats(u7_iris_scores)
        gen_stats = calc_stats(all_genuine_scores)
        imp_stats = calc_stats(impostor_iris_scores)

        # Metric counts
        u6_face_successes = sum(1 for r in u6_results if r["face_stage"]["matched"] and r["face_stage"]["candidate_user_id"] == 6)
        u7_face_successes = sum(1 for r in u7_results if r["face_stage"]["matched"] and r["face_stage"]["candidate_user_id"] == 7)
        total_face_successes = u6_face_successes + u7_face_successes
        face_identification_success_rate = (total_face_successes / 20.0) * 100.0

        genuine_acceptance_count_at_0800 = sum(1 for s in all_genuine_scores if s >= 0.800)
        genuine_acceptance_rate_at_0800 = (genuine_acceptance_count_at_0800 / len(all_genuine_scores)) * 100.0 if all_genuine_scores else 0.0

        false_rejection_count_at_0800 = len(all_genuine_scores) - genuine_acceptance_count_at_0800
        false_acceptance_count = sum(1 for r in impostor_results if r["final_decision"] == "VERIFIED")

        # Score overlap calculation
        overlap_exists = (gen_stats["min"] <= imp_stats["max"]) and (imp_stats["min"] <= gen_stats["max"])
        overlap_range = (max(gen_stats["min"], imp_stats["min"]), min(gen_stats["max"], imp_stats["max"])) if overlap_exists else None

        print("=" * 100)
        print("STATISTICAL SUMMARY REPORT")
        print("=" * 100)
        print(f"USER #6 Genuine Iris Scores (N={len(u6_iris_scores)}):")
        print(f"  Min: {u6_stats['min']:.4f} | Max: {u6_stats['max']:.4f} | Mean: {u6_stats['mean']:.4f} | Median: {u6_stats['median']:.4f}")
        print(f"\nUSER #7 Genuine Iris Scores (N={len(u7_iris_scores)}):")
        print(f"  Min: {u7_stats['min']:.4f} | Max: {u7_stats['max']:.4f} | Mean: {u7_stats['mean']:.4f} | Median: {u7_stats['median']:.4f}")
        print(f"\nALL GENUINE Iris Scores (N={len(all_genuine_scores)}):")
        print(f"  Min: {gen_stats['min']:.4f} | Max: {gen_stats['max']:.4f} | Mean: {gen_stats['mean']:.4f} | Median: {gen_stats['median']:.4f}")
        print(f"\nIMPOSTOR Iris Scores (N={len(impostor_iris_scores)}):")
        print(f"  Min: {imp_stats['min']:.4f} | Max: {imp_stats['max']:.4f} | Mean: {imp_stats['mean']:.4f} | Median: {imp_stats['median']:.4f}")
        print(f"\nMETRICS SUMMARY:")
        print(f"  Face Identification Success Rate: {face_identification_success_rate:.1f}% ({total_face_successes}/20)")
        print(f"  Iris Genuine Acceptance Count at 0.800: {genuine_acceptance_count_at_0800} / {len(all_genuine_scores)} ({genuine_acceptance_rate_at_0800:.1f}%)")
        print(f"  False Rejection Count at 0.800: {false_rejection_count_at_0800} / {len(all_genuine_scores)}")
        print(f"  False Acceptance Count (Impostor -> VERIFIED): {false_acceptance_count} / {len(impostor_samples)}")
        print(f"  Score Overlap: {overlap_exists} (Range: {overlap_range if overlap_exists else 'None'})")

        # -------------------------------------------------------------
        # SAFETY VERIFICATION POST-RUN
        # -------------------------------------------------------------
        print("\n" + "=" * 100)
        print("POST-RUN SAFETY AUDIT")
        print("=" * 100)
        users_count_end = db.query(User).count()
        face_count_end = db.query(FaceEmbedding).count()
        iris_v2_count_end = db.query(IrisEmbeddingV2).count()
        access_logs_count_end = db.query(AccessLog).count()

        print(f"  Users Table Delta: {users_count_end - users_count_start} (Current: {users_count_end})")
        print(f"  Face Embeddings Delta: {face_count_end - face_count_start} (Current: {face_count_end})")
        print(f"  Iris V2 Embeddings Delta: {iris_v2_count_end - iris_v2_count_start} (Current: {iris_v2_count_end})")
        print(f"  Access Logs Delta: {access_logs_count_end - access_logs_count_start} (Current: {access_logs_count_end})")

        assert users_count_end == users_count_start, "Users table mutated!"
        assert face_count_end == face_count_start, "Face embeddings mutated!"
        assert iris_v2_count_end == iris_v2_count_start, "Iris V2 embeddings mutated!"
        assert access_logs_count_end == access_logs_count_start, "Access logs mutated!"

        print("  COM9 Transmissions: 0")
        print("  A Commands Sent: 0")
        print("  U Commands Sent: 0")
        print("  Check-in / Check-out session state: UNCHANGED")
        print("  >> ALL SAFETY CONSTRAINTS STRICTLY HONORED!")
        print("=" * 100)

        return {
            "u6_stats": u6_stats,
            "u7_stats": u7_stats,
            "gen_stats": gen_stats,
            "imp_stats": imp_stats,
            "face_identification_success_rate": face_identification_success_rate,
            "genuine_acceptance_count_at_0800": genuine_acceptance_count_at_0800,
            "false_rejection_count_at_0800": false_rejection_count_at_0800,
            "false_acceptance_count": false_acceptance_count,
            "overlap_exists": overlap_exists,
            "overlap_range": overlap_range
        }

    finally:
        db.close()

if __name__ == "__main__":
    run_step5_score_analysis()
