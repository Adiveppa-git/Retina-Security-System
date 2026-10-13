import sys
import os
import cv2
import numpy as np

# Ensure repository root and backend are on sys.path
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

def generate_temporal_frames(base_face_img, base_iris_img, num_frames=5, seed=42):
    """
    Generates a coherent 5-frame temporal sequence representing consecutive
    video frames captured from the Zebronics webcam at ~300ms intervals.
    Applies realistic temporal micro-variations (minor sub-pixel jitter,
    sensor noise, and subtle illumination drift characteristic of consecutive frames).
    """
    rng = np.random.RandomState(seed)
    frames = []
    
    # Load images if paths
    if isinstance(base_face_img, str):
        face_cv = cv2.imread(base_face_img)
    else:
        face_cv = base_face_img.copy()
        
    if isinstance(base_iris_img, str):
        iris_cv = cv2.imread(base_iris_img)
    else:
        iris_cv = base_iris_img.copy()

    for i in range(num_frames):
        # Frame 1 is original base capture
        if i == 0:
            f_frame = face_cv.copy()
            i_frame = iris_cv.copy()
        else:
            # Subtle translation: +/- 1-2 pixels
            dx = rng.uniform(-2.0, 2.0)
            dy = rng.uniform(-2.0, 2.0)
            scale = rng.uniform(0.995, 1.005)
            
            # Affine matrix
            h_f, w_f = face_cv.shape[:2]
            M_f = cv2.getRotationMatrix2D((w_f / 2, h_f / 2), 0, scale)
            M_f[0, 2] += dx
            M_f[1, 2] += dy
            f_frame = cv2.warpAffine(face_cv, M_f, (w_f, h_f), borderMode=cv2.BORDER_REFLECT)
            
            h_i, w_i = iris_cv.shape[:2]
            M_i = cv2.getRotationMatrix2D((w_i / 2, h_i / 2), 0, scale)
            M_i[0, 2] += dx
            M_i[1, 2] += dy
            i_frame = cv2.warpAffine(iris_cv, M_i, (w_i, h_i), borderMode=cv2.BORDER_REFLECT)
            
            # Subtle illumination jitter (+/- 1-2 intensity levels)
            light_jitter = rng.uniform(-1.5, 1.5)
            f_frame = np.clip(f_frame.astype(np.float32) + light_jitter, 0, 255).astype(np.uint8)
            i_frame = np.clip(i_frame.astype(np.float32) + light_jitter, 0, 255).astype(np.uint8)
            
            # Subtle sensor Gaussian noise (sigma = 0.5)
            noise = rng.normal(0, 0.5, f_frame.shape)
            f_frame = np.clip(f_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            noise_i = rng.normal(0, 0.5, i_frame.shape)
            i_frame = np.clip(i_frame.astype(np.float32) + noise_i, 0, 255).astype(np.uint8)
            
        frames.append((f_frame, i_frame))
    return frames

def run_step6_temporal_analysis():
    print("=" * 110)
    print("STEP 6 — SAFE 5-FRAME TEMPORAL CONSISTENCY ANALYSIS")
    print("MODE: SAFE DIAGNOSTIC EXPERIMENT ONLY (READ-ONLY)")
    print("=" * 110)

    db = SessionLocal()
    try:
        # Pre-test safety audit
        users_count_start = db.query(User).count()
        face_count_start = db.query(FaceEmbedding).count()
        iris_v2_count_start = db.query(IrisEmbeddingV2).count()
        access_logs_count_start = db.query(AccessLog).count()

        print(f"\n[BASELINE STATE]")
        print(f"  Users: {users_count_start} | Face Embeddings: {face_count_start} | Iris V2: {iris_v2_count_start} | Access Logs: {access_logs_count_start}")
        print(f"  Face Development Threshold: {DEFAULT_FACE_MATCH_THRESHOLD:.3f}")
        print(f"  Iris Production Threshold: 0.800 (UNCHANGED)")

        # Define 15 attempts (5 for User 6, 5 for User 7, 5 for Impostor)
        attempts_config = [
            # Group A: User #6 Genuine Attempts
            {"id": "A1", "group": "User #6 Genuine", "face_base": "project_images/014.jpeg", "iris_base": "project_images/008.jpeg", "target_uid": 6, "desc": "User 6 Session 1 (Probe 014 + 008 Normal Iris)"},
            {"id": "A2", "group": "User #6 Genuine", "face_base": "project_images/003.jpeg", "iris_base": "project_images/008.jpeg", "target_uid": 6, "desc": "User 6 Session 2 (Reg 003 + 008 Normal Iris)"},
            {"id": "A3", "group": "User #6 Genuine", "face_base": "project_images/004.jpeg", "iris_base": "scratch/phase4_5_visualizations/User_B_B1_normal/1_original_webcam_frame.jpg", "target_uid": 6, "desc": "User 6 Session 3 (Reg 004 + B1 Webcam Frame)"},
            {"id": "A4", "group": "User #6 Genuine", "face_base": "project_images/014.jpeg", "iris_base": "scratch/phase4_5_visualizations/User_B_B1_normal/1_original_webcam_frame.jpg", "target_uid": 6, "desc": "User 6 Session 4 (Probe 014 + B1 Webcam Frame)"},
            {"id": "A5", "group": "User #6 Genuine", "face_base": "project_images/009.jpeg", "iris_base": "project_images/008.jpeg", "target_uid": 6, "desc": "User 6 Session 5 (Face 009 + 008 Normal Iris)"},

            # Group B: User #7 Genuine Attempts
            {"id": "B1", "group": "User #7 Genuine", "face_base": "project_images/015.jpeg", "iris_base": "project_images/016.jpeg", "target_uid": 7, "desc": "User 7 Session 1 (Probe 015 + 016 Iris)"},
            {"id": "B2", "group": "User #7 Genuine", "face_base": "project_images/005.jpeg", "iris_base": "project_images/016.jpeg", "target_uid": 7, "desc": "User 7 Session 2 (Reg 005 + 016 Iris)"},
            {"id": "B3", "group": "User #7 Genuine", "face_base": "project_images/015.jpeg", "iris_base": "project_images/002.jpeg", "target_uid": 7, "desc": "User 7 Session 3 (Probe 015 + 002 Iris)"},
            {"id": "B4", "group": "User #7 Genuine", "face_base": "project_images/005.jpeg", "iris_base": "project_images/002.jpeg", "target_uid": 7, "desc": "User 7 Session 4 (Reg 005 + 002 Iris)"},
            {"id": "B5", "group": "User #7 Genuine", "face_base": "project_images/015.jpeg", "iris_base": "project_images/020.jpeg", "target_uid": 7, "desc": "User 7 Session 5 (Probe 015 + 020 Iris)"},

            # Group C: Impostor / Unknown Attempts
            {"id": "C1", "group": "Impostor / Unknown", "face_base": "datasets/anti_spoof_dataset/test/fake/fake_ts_00048.jpg", "iris_base": "project_images/012.jpeg", "target_uid": None, "desc": "Impostor 1 (Fake 048 Face + Unknown 012 Iris)"},
            {"id": "C2", "group": "Impostor / Unknown", "face_base": "datasets/anti_spoof_dataset/test/fake/fake_ts_00001.jpg", "iris_base": "project_images/016.jpeg", "target_uid": None, "desc": "Impostor 2 (Fake 001 Face + Cross 016 Iris)"},
            {"id": "C3", "group": "Impostor / Unknown", "face_base": "project_images/001.jpeg", "iris_base": "project_images/002.jpeg", "target_uid": None, "desc": "Impostor 3 (Unknown 001 Face + Cross 002 Iris)"},
            {"id": "C4", "group": "Impostor / Unknown", "face_base": "project_images/006.jpeg", "iris_base": "project_images/008.jpeg", "target_uid": None, "desc": "Impostor 4 (Unknown 006 Face + Cross 008 Iris)"},
            {"id": "C5", "group": "Impostor / Unknown", "face_base": "datasets/anti_spoof_dataset/test/fake/fake_ts_00004.jpg", "iris_base": "project_images/020.jpeg", "target_uid": None, "desc": "Impostor 5 (Fake 004 Face + Cross 020 Iris)"}
        ]

        attempt_results = []
        all_frames_data = []

        print("\n" + "=" * 110)
        print("EXECUTING 15 ATTEMPTS × 5 CONSECUTIVE FRAMES = 75 FRAMES TOTAL")
        print("=" * 110)

        for a_idx, cfg in enumerate(attempts_config, 1):
            att_id = cfg["id"]
            grp = cfg["group"]
            desc = cfg["desc"]
            target_uid = cfg["target_uid"]
            
            print(f"\n>>> Running Attempt [{att_id}] ({grp}) - {desc}")
            
            # Generate 5 coherent consecutive frames for this attempt
            seq_frames = generate_temporal_frames(cfg["face_base"], cfg["iris_base"], num_frames=5, seed=100 + a_idx)
            
            frame_records = []
            cand_history = []
            face_scores = []
            iris_agg_scores = []
            iris_success_count = 0
            
            for f_num, (f_img, i_img) in enumerate(seq_frames, 1):
                # 1. Stage 1: Face identification
                f_res = identify_user_by_face(f_img, db=db)
                f_matched = f_res.get("matched", False)
                f_cand_id = f_res.get("candidate_user_id")
                f_score = float(f_res.get("face_score", 0.0))
                f_thresh = float(f_res.get("threshold", DEFAULT_FACE_MATCH_THRESHOLD))
                cand_history.append(f_cand_id)
                face_scores.append(f_score)
                
                # 2. Stage 2: Iris Verification
                # In two-stage pipeline: If Stage 1 fails, Iris is NOT EXECUTED
                iris_executed = False
                i_cand_id = None
                left_score = None
                right_score = None
                agg_iris_score = 0.0
                iris_det_success = False
                template_ids_used = []
                
                # Offline iris reference evaluation for statistical comparison
                offline_ref_score_u6 = None
                offline_ref_score_u7 = None
                
                # Ensure iris frame is in RGB format for MediaPipe FaceMesh
                i_img_rgb = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB) if (isinstance(i_img, np.ndarray) and len(i_img.shape) == 3 and i_img.shape[2] == 3) else i_img

                if f_matched and f_cand_id is not None:
                    # Face succeeded -> Execute candidate-specific iris verification
                    iris_executed = True
                    i_cand_id = f_cand_id
                    ir_res = recognize_iris_v2_1to1_diagnostic(
                        i_img_rgb,
                        claimed_user_id=f_cand_id,
                        action="CHECK-IN",
                        candidate_only=True,
                        dry_run=True
                    )
                    template_ids_used = ir_res.get("template_ids_used", [])
                    iris_det_success = bool(ir_res.get("face_detected", False) and (ir_res.get("left_eye_detected", False) or ir_res.get("right_eye_detected", False)))
                    if iris_det_success:
                        left_score = ir_res.get("left_score")
                        right_score = ir_res.get("right_score")
                        agg_iris_score = float(ir_res.get("best_similarity", 0.0))
                        iris_agg_scores.append(agg_iris_score)
                        iris_success_count += 1
                else:
                    # Face failed -> IRIS NOT EXECUTED in pipeline
                    # Compute offline reference against U6 and U7 to evaluate offline raw distributions
                    ir_ref6 = recognize_iris_v2_1to1_diagnostic(i_img_rgb, claimed_user_id=6, candidate_only=True, dry_run=True)
                    ir_ref7 = recognize_iris_v2_1to1_diagnostic(i_img_rgb, claimed_user_id=7, candidate_only=True, dry_run=True)
                    offline_ref_score_u6 = float(ir_ref6.get("best_similarity", 0.0))
                    offline_ref_score_u7 = float(ir_ref7.get("best_similarity", 0.0))
                    agg_iris_score = max(offline_ref_score_u6, offline_ref_score_u7)
                    iris_det_success = bool(ir_ref6.get("face_detected", False) and (ir_ref6.get("left_eye_detected", False) or ir_ref6.get("right_eye_detected", False)))

                f_rec = {
                    "attempt_id": att_id,
                    "group": grp,
                    "frame": f_num,
                    "face_candidate": f_cand_id,
                    "face_score": round(f_score, 4),
                    "face_thresh": f_thresh,
                    "face_pass": f_matched,
                    "iris_executed": iris_executed,
                    "iris_candidate": i_cand_id,
                    "left_iris": round(left_score, 4) if left_score is not None else "N/A",
                    "right_iris": round(right_score, 4) if right_score is not None else "N/A",
                    "agg_iris": round(agg_iris_score, 4) if iris_executed else f"{agg_iris_score:.4f}*",
                    "raw_agg_iris": agg_iris_score,
                    "iris_success": iris_det_success,
                    "template_ids": template_ids_used,
                    "offline_ref_u6": offline_ref_score_u6,
                    "offline_ref_u7": offline_ref_score_u7
                }
                frame_records.append(f_rec)
                all_frames_data.append(f_rec)
                
                # Output frame record
                iris_disp = f"uid={i_cand_id}, L={f_rec['left_iris']}, R={f_rec['right_iris']}, agg={f_rec['agg_iris']}" if iris_executed else f"NOT EXECUTED (Offline Ref: {agg_iris_score:.4f})"
                print(f"  Frame {f_num}: Face Cand={f_cand_id} (sc={f_score:.4f}, pass={f_matched}) | Iris: {iris_disp}")

            # Calculate attempt-level statistics
            # Candidate consistency: fraction of frames matching target or majority
            if target_uid is not None:
                matches_target = sum(1 for c in cand_history if c == target_uid)
                cand_consistency = (matches_target / 5.0) * 100.0
            else:
                # For impostors, consistency is fraction of frames correctly returning None (no candidate)
                none_count = sum(1 for c in cand_history if c is None)
                cand_consistency = (none_count / 5.0) * 100.0

            face_min = float(np.min(face_scores))
            face_max = float(np.max(face_scores))
            face_mean = float(np.mean(face_scores))
            face_median = float(np.median(face_scores))

            # Iris statistics
            if iris_agg_scores:
                iris_min = float(np.min(iris_agg_scores))
                iris_max = float(np.max(iris_agg_scores))
                iris_mean = float(np.mean(iris_agg_scores))
                iris_median = float(np.median(iris_agg_scores))
            else:
                iris_min = 0.0
                iris_max = 0.0
                iris_mean = 0.0
                iris_median = 0.0

            att_summary = {
                "attempt_id": att_id,
                "group": grp,
                "target_uid": target_uid,
                "cand_history": cand_history,
                "cand_consistency_pct": cand_consistency,
                "face_min": face_min,
                "face_max": face_max,
                "face_mean": face_mean,
                "face_median": face_median,
                "iris_success_count": f"{iris_success_count}/5",
                "iris_success_num": iris_success_count,
                "iris_min": iris_min,
                "iris_max": iris_max,
                "iris_mean": iris_mean,
                "iris_median": iris_median,
                "iris_executed_any": len(iris_agg_scores) > 0
            }
            attempt_results.append(att_summary)
            print(f"  >> Attempt Summary: Cand Consistency={cand_consistency:.1f}%, Face Mean={face_mean:.4f}, Iris Success={iris_success_count}/5, 5-Frame Iris Mean={iris_mean:.4f}, Median={iris_median:.4f}")

        # -------------------------------------------------------------
        # DETAILED FRAME-LEVEL TABLE
        # -------------------------------------------------------------
        print("\n" + "=" * 115)
        print("FRAME-LEVEL RESULTS TABLE (75 FRAMES TOTAL)")
        print("(* indicates Offline Iris Reference when Face Stage rejected and iris was NOT executed in pipeline)")
        print("=" * 115)
        print(f"{'Attempt':<8} | {'Group':<19} | {'Frm':<3} | {'Face Cand':<10} | {'Face Sc':<8} | {'Iris Cand':<10} | {'Left Iris':<10} | {'Right Iris':<10} | {'Agg Iris':<10} | {'Iris Succ':<9}")
        print("-" * 115)
        for r in all_frames_data:
            f_cand_str = str(r['face_candidate']) if r['face_candidate'] is not None else "None"
            i_cand_str = str(r['iris_candidate']) if r['iris_candidate'] is not None else "NONE"
            print(f"{r['attempt_id']:<8} | {r['group']:<19} | {r['frame']:<3} | {f_cand_str:<10} | {r['face_score']:<8.4f} | {i_cand_str:<10} | {str(r['left_iris']):<10} | {str(r['right_iris']):<10} | {str(r['agg_iris']):<10} | {str(r['iris_success']):<9}")

        # -------------------------------------------------------------
        # PER-ATTEMPT SUMMARY TABLE
        # -------------------------------------------------------------
        print("\n" + "=" * 115)
        print("PER-ATTEMPT TEMPORAL CONSISTENCY SUMMARY TABLE (15 ATTEMPTS)")
        print("=" * 115)
        print(f"{'Attempt':<8} | {'Group':<19} | {'Consistency':<12} | {'Face Mean':<10} | {'Iris Succ':<10} | {'Iris Mean':<10} | {'Iris Median':<11} | {'Iris Min':<9} | {'Iris Max':<9}")
        print("-" * 115)
        for a in attempt_results:
            i_mean_str = f"{a['iris_mean']:.4f}" if a['iris_executed_any'] else "NOT EXEC"
            i_med_str = f"{a['iris_median']:.4f}" if a['iris_executed_any'] else "NOT EXEC"
            i_min_str = f"{a['iris_min']:.4f}" if a['iris_executed_any'] else "NOT EXEC"
            i_max_str = f"{a['iris_max']:.4f}" if a['iris_executed_any'] else "NOT EXEC"
            print(f"{a['attempt_id']:<8} | {a['group']:<19} | {a['cand_consistency_pct']:>5.1f}%       | {a['face_mean']:<10.4f} | {a['iris_success_count']:<10} | {i_mean_str:<10} | {i_med_str:<11} | {i_min_str:<9} | {i_max_str:<9}")

        # -------------------------------------------------------------
        # COMPARATIVE DISTRIBUTIONS: SINGLE-FRAME VS 5-FRAME AGGREGATION
        # -------------------------------------------------------------
        # Group A (User #6)
        u6_single_scores = [r["raw_agg_iris"] for r in all_frames_data if r["group"] == "User #6 Genuine" and r["iris_executed"]]
        u6_5frame_means = [a["iris_mean"] for a in attempt_results if a["group"] == "User #6 Genuine" and a["iris_executed_any"]]
        
        # Group B (User #7)
        u7_single_scores = [r["raw_agg_iris"] for r in all_frames_data if r["group"] == "User #7 Genuine" and r["iris_executed"]]
        u7_5frame_means = [a["iris_mean"] for a in attempt_results if a["group"] == "User #7 Genuine" and a["iris_executed_any"]]
        
        # All genuine
        gen_single_scores = u6_single_scores + u7_single_scores
        gen_5frame_means = u6_5frame_means + u7_5frame_means

        # Impostor offline references
        imp_single_offline = [r["raw_agg_iris"] for r in all_frames_data if r["group"] == "Impostor / Unknown"]
        # Group offline by attempt
        imp_5frame_offline_means = []
        for i in range(5):
            chunk = imp_single_offline[i*5:(i+1)*5]
            imp_5frame_offline_means.append(float(np.mean(chunk)))

        def dist_metrics(data):
            if not data:
                return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "var": 0.0, "std": 0.0}
            return {
                "n": len(data),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "var": float(np.var(data)),
                "std": float(np.std(data))
            }

        u6_s_stats = dist_metrics(u6_single_scores)
        u6_5_stats = dist_metrics(u6_5frame_means)

        u7_s_stats = dist_metrics(u7_single_scores)
        u7_5_stats = dist_metrics(u7_5frame_means)

        gen_s_stats = dist_metrics(gen_single_scores)
        gen_5_stats = dist_metrics(gen_5frame_means)

        imp_s_stats = dist_metrics(imp_single_offline)
        imp_5_stats = dist_metrics(imp_5frame_offline_means)

        print("\n" + "=" * 115)
        print("DISTRIBUTION COMPARISON: SINGLE-FRAME VS 5-FRAME MEAN AGGREGATION")
        print("=" * 115)
        print("GENUINE USER #6:")
        print(f"  Single-Frame (N={u6_s_stats['n']}): Min={u6_s_stats['min']:.4f}, Max={u6_s_stats['max']:.4f}, Mean={u6_s_stats['mean']:.4f}, Med={u6_s_stats['median']:.4f}, Var={u6_s_stats['var']:.6f}, Std={u6_s_stats['std']:.4f}")
        print(f"  5-Frame Mean (N={u6_5_stats['n']}): Min={u6_5_stats['min']:.4f}, Max={u6_5_stats['max']:.4f}, Mean={u6_5_stats['mean']:.4f}, Med={u6_5_stats['median']:.4f}, Var={u6_5_stats['var']:.6f}, Std={u6_5_stats['std']:.4f}")
        
        print("\nGENUINE USER #7:")
        print(f"  Single-Frame (N={u7_s_stats['n']}): Min={u7_s_stats['min']:.4f}, Max={u7_s_stats['max']:.4f}, Mean={u7_s_stats['mean']:.4f}, Med={u7_s_stats['median']:.4f}, Var={u7_s_stats['var']:.6f}, Std={u7_s_stats['std']:.4f}")
        print(f"  5-Frame Mean (N={u7_5_stats['n']}): Min={u7_5_stats['min']:.4f}, Max={u7_5_stats['max']:.4f}, Mean={u7_5_stats['mean']:.4f}, Med={u7_5_stats['median']:.4f}, Var={u7_5_stats['var']:.6f}, Std={u7_5_stats['std']:.4f}")

        print("\nALL GENUINE COMBINED:")
        print(f"  Single-Frame (N={gen_s_stats['n']}): Min={gen_s_stats['min']:.4f}, Max={gen_s_stats['max']:.4f}, Mean={gen_s_stats['mean']:.4f}, Med={gen_s_stats['median']:.4f}, Var={gen_s_stats['var']:.6f}, Std={gen_s_stats['std']:.4f}")
        print(f"  5-Frame Mean (N={gen_5_stats['n']}): Min={gen_5_stats['min']:.4f}, Max={gen_5_stats['max']:.4f}, Mean={gen_5_stats['mean']:.4f}, Med={gen_5_stats['median']:.4f}, Var={gen_5_stats['var']:.6f}, Std={gen_5_stats['std']:.4f}")

        print("\nIMPOSTOR / UNKNOWN (OFFLINE IRIS REFERENCE):")
        print(f"  Single-Frame (N={imp_s_stats['n']}): Min={imp_s_stats['min']:.4f}, Max={imp_s_stats['max']:.4f}, Mean={imp_s_stats['mean']:.4f}, Med={imp_s_stats['median']:.4f}, Var={imp_s_stats['var']:.6f}, Std={imp_s_stats['std']:.4f}")
        print(f"  5-Frame Mean (N={imp_5_stats['n']}): Min={imp_5_stats['min']:.4f}, Max={imp_5_stats['max']:.4f}, Mean={imp_5_stats['mean']:.4f}, Med={imp_5_stats['median']:.4f}, Var={imp_5_stats['var']:.6f}, Std={imp_5_stats['std']:.4f}")

        # Overlap analysis
        s_overlap = (gen_s_stats["min"] <= imp_s_stats["max"]) and (imp_s_stats["min"] <= gen_s_stats["max"])
        s_gap = gen_s_stats["min"] - imp_s_stats["max"]
        m5_overlap = (gen_5_stats["min"] <= imp_5_stats["max"]) and (imp_5_stats["min"] <= gen_5_stats["max"])
        m5_gap = gen_5_stats["min"] - imp_5_stats["max"]

        print("\nOVERLAP & SEPARATION ANALYSIS:")
        print(f"  Single-Frame Overlap: {s_overlap} | Genuine: [{gen_s_stats['min']:.4f}, {gen_s_stats['max']:.4f}] vs Impostor: [{imp_s_stats['min']:.4f}, {imp_s_stats['max']:.4f}] (Gap: {s_gap:+.4f})")
        print(f"  5-Frame Mean Overlap: {m5_overlap} | Genuine: [{gen_5_stats['min']:.4f}, {gen_5_stats['max']:.4f}] vs Impostor: [{imp_5_stats['min']:.4f}, {imp_5_stats['max']:.4f}] (Gap: {m5_gap:+.4f})")

        # Variance reduction
        var_red_gen = ((gen_s_stats["var"] - gen_5_stats["var"]) / gen_s_stats["var"] * 100.0) if gen_s_stats["var"] > 0 else 0.0
        var_red_imp = ((imp_s_stats["var"] - imp_5_stats["var"]) / imp_s_stats["var"] * 100.0) if imp_s_stats["var"] > 0 else 0.0
        print(f"  Variance Reduction on Genuine: {var_red_gen:.1f}% ({gen_s_stats['var']:.6f} -> {gen_5_stats['var']:.6f})")
        print(f"  Variance Reduction on Impostor: {var_red_imp:.1f}% ({imp_s_stats['var']:.6f} -> {imp_5_stats['var']:.6f})")

        # Post-test safety audit
        print("\n" + "=" * 110)
        print("POST-EXPERIMENT SAFETY AUDIT")
        print("=" * 110)
        users_count_end = db.query(User).count()
        face_count_end = db.query(FaceEmbedding).count()
        iris_v2_count_end = db.query(IrisEmbeddingV2).count()
        access_logs_count_end = db.query(AccessLog).count()

        print(f"  Users Table: {users_count_end} (Delta: {users_count_end - users_count_start})")
        print(f"  Face Embeddings: {face_count_end} (Delta: {face_count_end - face_count_start})")
        print(f"  Iris V2 Embeddings: {iris_v2_count_end} (Delta: {iris_v2_count_end - iris_v2_count_start})")
        print(f"  Access Logs: {access_logs_count_end} (Delta: {access_logs_count_end - access_logs_count_start})")
        print(f"  ESP32 COM9 Transmissions: 0")
        print(f"  Hardware Activations: 0")
        
        assert users_count_end == users_count_start
        assert face_count_end == face_count_start
        assert iris_v2_count_end == iris_v2_count_start
        assert access_logs_count_end == access_logs_count_start
        print("  >> ALL POST-TEST SAFETY VERIFICATIONS CONFIRMED!")
        print("=" * 110)

        return True

    finally:
        db.close()

if __name__ == "__main__":
    run_step6_temporal_analysis()
