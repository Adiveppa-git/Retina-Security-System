import requests
import os
import glob
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000"


def run_full_security_suite():
    print("==================================================================")
    print(" EXECUTING COMPLETE BIOMETRIC SECURITY VERIFICATION TEST SUITE")
    print("==================================================================")

    test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
    subjs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))]

    if len(subjs) < 2:
        print("[ERROR] Need at least 2 subject directories in test dataset.")
        return

    subj1_imgs = glob.glob(os.path.join(test_rec_dir, subjs[0], "*.jpg"))
    subj2_imgs = glob.glob(os.path.join(test_rec_dir, subjs[1], "*.jpg"))

    name_user1 = f"Registered_User_1_{subjs[0]}"
    name_user2 = f"Registered_User_2_{subjs[1]}"

    # Reset DB before starting tests
    import sys
    sys.path.append(os.path.abspath("."))
    from scratch.clear_db_test_records import clear_database
    clear_database()

    # Step 1: Register User 1 with 2 templates
    with open(subj1_imgs[0], "rb") as f1, open(subj1_imgs[1], "rb") as f2:
        files = [
            ("files", (os.path.basename(subj1_imgs[0]), f1, "image/jpeg")),
            ("files", (os.path.basename(subj1_imgs[1]), f2, "image/jpeg"))
        ]
        reg1_res = requests.post(f"{API_URL}/user/add", data={"name": name_user1, "email": "user1@example.com"}, files=files)
    print(f"\n[REGISTRATION 1] Output: {reg1_res.json()}")

    # Step 2: Register User 2 with 1 template
    with open(subj2_imgs[0], "rb") as f:
        files = [("files", (os.path.basename(subj2_imgs[0]), f, "image/jpeg"))]
        reg2_res = requests.post(f"{API_URL}/user/add", data={"name": name_user2, "email": "user2@example.com"}, files=files)
    print(f"[REGISTRATION 2] Output: {reg2_res.json()}")

    # -------------------------------------------------------------
    # TEST A — Registered Person -> ACCESS_GRANTED
    # -------------------------------------------------------------
    with open(subj1_imgs[2], "rb") as f:
        res_a = requests.post(f"{API_URL}/iris/verify", files={"file": f}).json()

    print("\n--- TEST A: Registered Person ---")
    print(f"Final Decision : {res_a.get('final_decision')}")
    print(f"Decision Reason: {res_a.get('decision_reason')}")
    print(f"Best Match User: '{res_a.get('best_match_user_name')}' (ID #{res_a.get('best_match_user_id')})")
    print(f"Best Similarity: {res_a.get('best_similarity')}")
    print(f"Hardware State : Door={res_a.get('hardware', {}).get('door')}, LED={res_a.get('hardware', {}).get('led')}")
    assert res_a.get('final_decision') == 'ACCESS_GRANTED', "TEST A FAILED!"
    print(">>> TEST A PASSED: Registered Person Granted Access!")

    # -------------------------------------------------------------
    # TEST B — Completely Unregistered Person -> ACCESS_DENIED
    # -------------------------------------------------------------
    if len(subjs) >= 3:
        subj3_imgs = glob.glob(os.path.join(test_rec_dir, subjs[2], "*.jpg"))
        with open(subj3_imgs[0], "rb") as f:
            res_b = requests.post(f"{API_URL}/iris/verify", files={"file": f}).json()

        print("\n--- TEST B: Completely Unregistered Person ---")
        print(f"Final Decision : {res_b.get('final_decision')}")
        print(f"Decision Reason: {res_b.get('decision_reason')}")
        print(f"Best Match User: '{res_b.get('best_match_user_name')}'")
        print(f"Best Similarity: {res_b.get('best_similarity')}")
        print(f"Hardware State : Door={res_b.get('hardware', {}).get('door')}, LED={res_b.get('hardware', {}).get('led')}")
        assert res_b.get('final_decision') == 'ACCESS_DENIED', "TEST B FAILED!"
        print(">>> TEST B PASSED: Unregistered Person Denied Access!")

    # -------------------------------------------------------------
    # TEST D — No Face (Solid Black Image) -> ACCESS_DENIED
    # -------------------------------------------------------------
    black_img = Image.fromarray(np.zeros((320, 440, 3), dtype=np.uint8))
    black_img.save("scratch/test_black.jpg")
    with open("scratch/test_black.jpg", "rb") as f:
        res_d = requests.post(f"{API_URL}/iris/verify", files={"file": f}).json()

    print("\n--- TEST D: No Face Frame ---")
    print(f"Face Detected  : {res_d.get('face_detected')}")
    print(f"Final Decision : {res_d.get('final_decision')}")
    print(f"Decision Reason: {res_d.get('decision_reason')}")
    print(f"Hardware State : Door={res_d.get('hardware', {}).get('door')}, LED={res_d.get('hardware', {}).get('led')}")
    assert res_d.get('final_decision') == 'ACCESS_DENIED', "TEST D FAILED!"
    print(">>> TEST D PASSED: No Face Frame Denied Access!")

    print("\n==================================================================")
    print(" ALL BIOMETRIC SECURITY TEST SUITES PASSED 100% PERFECTLY! ")
    print("==================================================================")


if __name__ == "__main__":
    run_full_security_suite()
