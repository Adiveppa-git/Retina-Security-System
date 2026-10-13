import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project imports resolve
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace, "backend"))
sys.path.insert(0, workspace)

from app.services.iris_detection_service import detect_and_crop_iris
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

def extract_features_both_pipelines(img_path):
    """
    Extracts eye crops and 512-D embeddings for both:
    - Pipeline A: Baseline padded eye bbox (left_eye_bbox, right_eye_bbox)
    - Pipeline B: Experimental tight square iris crop (R = round(1.25 * iris_radius))
    """
    if not os.path.exists(img_path):
        return None, f"FILE_MISSING: {img_path}"

    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    res = detect_and_crop_iris(img_np)
    pl = res.payload

    if not pl.get("face_detected"):
        return None, f"NO_FACE_DETECTED: {pl.get('failure_reason')}"

    # Pipeline A: Baseline Crops
    feat_a = {"L": None, "R": None, "L_box": None, "R_box": None}
    if pl.get("left_iris_detected") and pl.get("left_eye_bbox"):
        lx1, ly1, lx2, ly2 = pl["left_eye_bbox"]
        crop_a_l = img_np[ly1:ly2, lx1:lx2]
        if crop_a_l.size > 0:
            feat_a["L"] = np.array(extract_v2_embedding(Image.fromarray(crop_a_l)), dtype=np.float32)
            feat_a["L_box"] = (lx2 - lx1, ly2 - ly1)

    if pl.get("right_iris_detected") and pl.get("right_eye_bbox"):
        rx1, ry1, rx2, ry2 = pl["right_eye_bbox"]
        crop_a_r = img_np[ry1:ry2, rx1:rx2]
        if crop_a_r.size > 0:
            feat_a["R"] = np.array(extract_v2_embedding(Image.fromarray(crop_a_r)), dtype=np.float32)
            feat_a["R_box"] = (rx2 - rx1, ry2 - ry1)

    # Pipeline B: Tight Iris Crops
    feat_b = {"L": None, "R": None, "L_box": None, "R_box": None, "L_r": None, "R_r": None}
    if pl.get("left_iris_detected") and pl.get("left_pupil_center") is not None:
        left_cx, left_cy = pl["left_pupil_center"]
        left_r = pl.get("left_iris_radius", 0.0)
        l_R = int(round(1.25 * left_r))
        t_lx1 = max(0, left_cx - l_R)
        t_ly1 = max(0, left_cy - l_R)
        t_lx2 = min(w, left_cx + l_R)
        t_ly2 = min(h, left_cy + l_R)
        crop_b_l = img_np[t_ly1:t_ly2, t_lx1:t_lx2]
        if crop_b_l.size > 0:
            feat_b["L"] = np.array(extract_v2_embedding(Image.fromarray(crop_b_l)), dtype=np.float32)
            feat_b["L_box"] = (t_lx2 - t_lx1, t_ly2 - t_ly1)
            feat_b["L_r"] = left_r

    if pl.get("right_iris_detected") and pl.get("right_pupil_center") is not None:
        right_cx, right_cy = pl["right_pupil_center"]
        right_r = pl.get("right_iris_radius", 0.0)
        r_R = int(round(1.25 * right_r))
        t_rx1 = max(0, right_cx - r_R)
        t_ry1 = max(0, right_cy - r_R)
        t_rx2 = min(w, right_cx + r_R)
        t_ry2 = min(h, right_cy + r_R)
        crop_b_r = img_np[t_ry1:t_ry2, t_rx1:t_rx2]
        if crop_b_r.size > 0:
            feat_b["R"] = np.array(extract_v2_embedding(Image.fromarray(crop_b_r)), dtype=np.float32)
            feat_b["R_box"] = (t_rx2 - t_rx1, t_ry2 - t_ry1)
            feat_b["R_r"] = right_r

    return {"A": feat_a, "B": feat_b, "payload": pl}, "OK"

def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def compute_combined_score(sim_l, sim_r):
    if sim_l is not None and sim_r is not None:
        return (sim_l + sim_r) / 2.0
    elif sim_l is not None:
        return sim_l
    elif sim_r is not None:
        return sim_r
    else:
        return None

def run_ab_experiment():
    print("=" * 80)
    print("CONTROLLED OFFLINE A/B EXPERIMENT: BASELINE vs TIGHT IRIS CROP")
    print("=" * 80)

    # 1. Verify Dataset Files
    ref_files = [
        os.path.join(workspace, "project_images", "008.jpeg"),
        os.path.join(workspace, "project_images", "018.jpeg")
    ]
    genuine_files = [
        os.path.join(workspace, "scratch", f"live_sample_{i}.jpg") for i in range(1, 11)
    ]
    impostor_files = [
        os.path.join(workspace, "project_images", "002.jpeg"),
        os.path.join(workspace, "project_images", "020.jpeg"),
        os.path.join(workspace, "project_images", "012.jpeg"),
        os.path.join(workspace, "project_images", "016.jpeg")
    ]

    all_files = ref_files + genuine_files + impostor_files
    print("\n--- 1. VERIFYING DATA FILES EXISTENCE ---")
    all_exist = True
    for f in all_files:
        exists = os.path.exists(f)
        rel = os.path.relpath(f, workspace)
        print(f"  {rel:<35}: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("ERROR: One or more required test files are missing. Aborting.")
        return

    # 2. Extract Reference / Enrollment Templates
    print("\n--- 2. EXTRACTING ENROLLMENT / REFERENCE TEMPLATES ---")
    ref_templates = {"A": {"L": [], "R": []}, "B": {"L": [], "R": []}}

    for rf in ref_files:
        bname = os.path.basename(rf)
        feats, status = extract_features_both_pipelines(rf)
        if status != "OK":
            print(f"  Reference {bname}: FAILED ({status})")
            continue
        print(f"  Reference {bname}: SUCCESS")
        print(f"    Baseline Crop Sizes -> L: {feats['A']['L_box']}, R: {feats['A']['R_box']}")
        print(f"    Tight Crop Sizes    -> L: {feats['B']['L_box']} (r={feats['B']['L_r']}), R: {feats['B']['R_box']} (r={feats['B']['R_r']})")

        if feats["A"]["L"] is not None:
            ref_templates["A"]["L"].append((bname, feats["A"]["L"]))
        if feats["A"]["R"] is not None:
            ref_templates["A"]["R"].append((bname, feats["A"]["R"]))
        if feats["B"]["L"] is not None:
            ref_templates["B"]["L"].append((bname, feats["B"]["L"]))
        if feats["B"]["R"] is not None:
            ref_templates["B"]["R"].append((bname, feats["B"]["R"]))

    print(f"Enrolled Template Counts:")
    print(f"  Pipeline A (Baseline) -> Left Templates: {len(ref_templates['A']['L'])}, Right Templates: {len(ref_templates['A']['R'])}")
    print(f"  Pipeline B (Tight)    -> Left Templates: {len(ref_templates['B']['L'])}, Right Templates: {len(ref_templates['B']['R'])}")

    if not ref_templates["A"]["L"] and not ref_templates["A"]["R"]:
        print("ERROR: No valid enrollment templates could be extracted. Aborting.")
        return

    # Helper to evaluate query against reference templates
    def evaluate_query(query_feats, pipeline_key):
        q = query_feats[pipeline_key]
        sim_l = None
        if q["L"] is not None and ref_templates[pipeline_key]["L"]:
            l_sims = [cosine_sim(q["L"], ref_vec) for _, ref_vec in ref_templates[pipeline_key]["L"]]
            sim_l = max(l_sims)

        sim_r = None
        if q["R"] is not None and ref_templates[pipeline_key]["R"]:
            r_sims = [cosine_sim(q["R"], ref_vec) for _, ref_vec in ref_templates[pipeline_key]["R"]]
            sim_r = max(r_sims)

        comb = compute_combined_score(sim_l, sim_r)
        return sim_l, sim_r, comb

    # 3. Evaluate Genuine Live Captures
    print("\n--- 3. EVALUATING GENUINE LIVE SAMPLES (10 FRAMES) ---")
    gen_results = []
    for gf in genuine_files:
        bname = os.path.basename(gf)
        feats, status = extract_features_both_pipelines(gf)
        if status != "OK":
            print(f"  {bname:<22}: Rejected by quality/face gate ({status})")
            continue

        l_a, r_a, comb_a = evaluate_query(feats, "A")
        l_b, r_b, comb_b = evaluate_query(feats, "B")

        gen_results.append({
            "name": bname,
            "A_L": l_a, "A_R": r_a, "A_comb": comb_a,
            "B_L": l_b, "B_R": r_b, "B_comb": comb_b,
            "A_box_L": feats["A"]["L_box"], "A_box_R": feats["A"]["R_box"],
            "B_box_L": feats["B"]["L_box"], "B_box_R": feats["B"]["R_box"],
        })

        l_a_s = f"{l_a:.4f}" if l_a is not None else "N/A"
        r_a_s = f"{r_a:.4f}" if r_a is not None else "N/A"
        c_a_s = f"{comb_a:.4f}" if comb_a is not None else "N/A"

        l_b_s = f"{l_b:.4f}" if l_b is not None else "N/A"
        r_b_s = f"{r_b:.4f}" if r_b is not None else "N/A"
        c_b_s = f"{comb_b:.4f}" if comb_b is not None else "N/A"

        print(f"  {bname:<20} | Base [L:{l_a_s} R:{r_a_s} -> Comb:{c_a_s}] | Tight [L:{l_b_s} R:{r_b_s} -> Comb:{c_b_s}]")

    # 4. Evaluate Impostor Captures
    print("\n--- 4. EVALUATING IMPOSTOR SAMPLES (4 DISTINCT SUBJECTS) ---")
    imp_results = []
    for imf in impostor_files:
        bname = os.path.basename(imf)
        feats, status = extract_features_both_pipelines(imf)
        if status != "OK":
            print(f"  {bname:<22}: Rejected ({status})")
            continue

        l_a, r_a, comb_a = evaluate_query(feats, "A")
        l_b, r_b, comb_b = evaluate_query(feats, "B")

        imp_results.append({
            "name": bname,
            "A_L": l_a, "A_R": r_a, "A_comb": comb_a,
            "B_L": l_b, "B_R": r_b, "B_comb": comb_b,
        })

        l_a_s = f"{l_a:.4f}" if l_a is not None else "N/A"
        r_a_s = f"{r_a:.4f}" if r_a is not None else "N/A"
        c_a_s = f"{comb_a:.4f}" if comb_a is not None else "N/A"

        l_b_s = f"{l_b:.4f}" if l_b is not None else "N/A"
        r_b_s = f"{r_b:.4f}" if r_b is not None else "N/A"
        c_b_s = f"{comb_b:.4f}" if comb_b is not None else "N/A"

        print(f"  {bname:<20} | Base [L:{l_a_s} R:{r_a_s} -> Comb:{c_a_s}] | Tight [L:{l_b_s} R:{r_b_s} -> Comb:{c_b_s}]")

    # 5. Statistical Calculations
    def compute_stats(scores):
        arr = np.array([s for s in scores if s is not None], dtype=np.float64)
        if len(arr) == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0
        return len(arr), float(np.mean(arr)), float(np.median(arr)), float(np.min(arr)), float(np.max(arr)), float(np.std(arr))

    gen_scores_a = [r["A_comb"] for r in gen_results if r["A_comb"] is not None]
    gen_scores_b = [r["B_comb"] for r in gen_results if r["B_comb"] is not None]

    imp_scores_a = [r["A_comb"] for r in imp_results if r["A_comb"] is not None]
    imp_scores_b = [r["B_comb"] for r in imp_results if r["B_comb"] is not None]

    n_g_a, mean_g_a, med_g_a, min_g_a, max_g_a, std_g_a = compute_stats(gen_scores_a)
    n_i_a, mean_i_a, med_i_a, min_i_a, max_i_a, std_i_a = compute_stats(imp_scores_a)

    n_g_b, mean_g_b, med_g_b, min_g_b, max_g_b, std_g_b = compute_stats(gen_scores_b)
    n_i_b, mean_i_b, med_i_b, min_i_b, max_i_b, std_i_b = compute_stats(imp_scores_b)

    overlap_a = max(0.0, max_i_a - min_g_a) if (min_g_a <= max_i_a) else 0.0
    overlap_b = max(0.0, max_i_b - min_g_b) if (min_g_b <= max_i_b) else 0.0

    margin_a = mean_g_a - mean_i_a
    margin_b = mean_g_b - mean_i_b

    print("\n" + "=" * 80)
    print("DETAILED STATISTICAL REPORT")
    print("=" * 80)

    print("\n[PIPELINE A — CURRENT BASELINE]")
    print(f"  - Number of Genuine Comparisons : {n_g_a}")
    print(f"  - Number of Impostor Comparisons: {n_i_a}")
    print(f"  - Genuine Mean                  : {mean_g_a:.4f}")
    print(f"  - Genuine Median                : {med_g_a:.4f}")
    print(f"  - Genuine Minimum               : {min_g_a:.4f}")
    print(f"  - Genuine Maximum               : {max_g_a:.4f}")
    print(f"  - Genuine Standard Deviation    : {std_g_a:.4f}")
    print(f"  - Impostor Mean                 : {mean_i_a:.4f}")
    print(f"  - Impostor Median               : {med_i_a:.4f}")
    print(f"  - Impostor Minimum              : {min_i_a:.4f}")
    print(f"  - Impostor Maximum              : {max_i_a:.4f}")
    print(f"  - Impostor Standard Deviation   : {std_i_a:.4f}")
    print(f"  - Lowest Genuine Score          : {min_g_a:.4f}")
    print(f"  - Highest Impostor Score        : {max_i_a:.4f}")
    print(f"  - Distributions Overlap?        : {'YES (Severe)' if min_g_a < max_i_a else 'NO'}")
    print(f"  - Overlap Extent                : {overlap_a:.4f}")
    print(f"  - Genuine Mean - Impostor Mean  : {margin_a:.4f}")

    print("\n[PIPELINE B — EXPERIMENTAL TIGHT CROP]")
    print(f"  - Number of Genuine Comparisons : {n_g_b}")
    print(f"  - Number of Impostor Comparisons: {n_i_b}")
    print(f"  - Genuine Mean                  : {mean_g_b:.4f}")
    print(f"  - Genuine Median                : {med_g_b:.4f}")
    print(f"  - Genuine Minimum               : {min_g_b:.4f}")
    print(f"  - Genuine Maximum               : {max_g_b:.4f}")
    print(f"  - Genuine Standard Deviation    : {std_g_b:.4f}")
    print(f"  - Impostor Mean                 : {mean_i_b:.4f}")
    print(f"  - Impostor Median               : {med_i_b:.4f}")
    print(f"  - Impostor Minimum              : {min_i_b:.4f}")
    print(f"  - Impostor Maximum              : {max_i_b:.4f}")
    print(f"  - Impostor Standard Deviation   : {std_i_b:.4f}")
    print(f"  - Lowest Genuine Score          : {min_g_b:.4f}")
    print(f"  - Highest Impostor Score        : {max_i_b:.4f}")
    print(f"  - Distributions Overlap?        : {'YES (Severe)' if min_g_b < max_i_b else 'NO'}")
    print(f"  - Overlap Extent                : {overlap_b:.4f}")
    print(f"  - Genuine Mean - Impostor Mean  : {margin_b:.4f}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY & COMPARISON")
    print("=" * 80)
    print("BASELINE:")
    print(f"  Genuine range : [{min_g_a:.4f}, {max_g_a:.4f}]")
    print(f"  Impostor range: [{min_i_a:.4f}, {max_i_a:.4f}]")
    print(f"  Overlap       : {overlap_a:.4f}")
    print(f"  Separation    : {margin_a:.4f}")
    print("\nTIGHT:")
    print(f"  Genuine range : [{min_g_b:.4f}, {max_g_b:.4f}]")
    print(f"  Impostor range: [{min_i_b:.4f}, {max_i_b:.4f}]")
    print(f"  Overlap       : {overlap_b:.4f}")
    print(f"  Separation    : {margin_b:.4f}")

if __name__ == "__main__":
    run_ab_experiment()
