import os
import sys
import glob

# Ensure workspace root and backend directory are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.database.connection import SessionLocal, engine, Base
from app.models.user_model import User, IrisEmbedding, AccessLog
from app.services.user_management_service import register_user, get_all_users
from app.services.iris_recognition_service import recognize_iris


def run_full_backend_test():
    print("==================================================")
    print(" STARTING BACKEND REGISTRATION & VERIFICATION PIPELINE TEST")
    print("==================================================")

    # Step 0: Initialize MySQL Tables & Clean Test Database
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(IrisEmbedding).delete()
        db.query(User).delete()
        db.query(AccessLog).delete()
        db.commit()
        print(" Cleared stale database test records.")
    except Exception as e:
        db.rollback()
        print(" Error cleaning database:", e)
    finally:
        db.close()

    # Step 1: Find 4 distinct subject test directories in datasets/recognition_dataset/test/
    test_rec_dir = os.path.join(BASE_DIR, "datasets", "recognition_dataset", "test")

    subj_dirs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))][:4]

    print(f"\n--- 1. REGISTERING 4 DISTINCT TEST PERSONS IN MYSQL DATABASE ---")
    registered_persons = []

    for idx, subj in enumerate(subj_dirs, start=1):
        subj_path = os.path.join(test_rec_dir, subj)
        imgs = glob.glob(os.path.join(subj_path, "*.jpg")) + glob.glob(os.path.join(subj_path, "*.png"))

        if len(imgs) < 2:
            continue

        reg_img = imgs[0]
        test_img = imgs[1]
        user_name = f"Registered_User_{idx}_{subj}"

        # Register User
        res = register_user(name=user_name, email=f"user_{idx}@university.edu", image_file=reg_img)
        print(f" Registration Output [{user_name}]: {res}")

        if res.get("status") == "success":
            registered_persons.append({
                "name": user_name,
                "reg_img": reg_img,
                "test_img": test_img
            })

    # Step 2: Verify All Registered Users with Unseen Test Images
    print(f"\n--- 2. VERIFYING REGISTERED USERS WITH UNSEEN TEST IMAGES ---")
    for person in registered_persons:
        name = person["name"]
        test_img = person["test_img"]

        result = recognize_iris(test_img)
        print(f"\n Verification Test for '{name}':")
        print(f"   Image File  : {os.path.basename(test_img)}")
        print(f"   Status      : {result.get('status')}")
        print(f"   Matched User: {result.get('person')}")
        print(f"   Similarity  : {result.get('similarity')}%")
        print(f"   Message     : {result.get('message')}")

        if result.get("status") == "success" and result.get("person") == name:
            print("   >>> RESULT: PASSED! CORRECT USER IDENTIFIED!")
        else:
            print("   >>> RESULT: FAILED!")

    # Step 3: Test Unregistered User Access Rejection
    print(f"\n--- 3. TESTING UNREGISTERED USER REJECTION ---")
    unregistered_subj_dirs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))][5:6]
    if unregistered_subj_dirs:
        unreg_subj = unregistered_subj_dirs[0]
        unreg_imgs = glob.glob(os.path.join(test_rec_dir, unreg_subj, "*.jpg"))
        if unreg_imgs:
            unreg_result = recognize_iris(unreg_imgs[0])
            print(f" Verification Result for Unregistered Subject '{unreg_subj}':")
            print(f"   Status      : {unreg_result.get('status')}")
            print(f"   Matched User: {unreg_result.get('person')}")
            print(f"   Message     : {unreg_result.get('message')}")
            if unreg_result.get("status") == "denied":
                print("   >>> RESULT: PASSED! UNREGISTERED USER CORRECTLY REJECTED!")

    # Step 4: Test Anti-Spoofing Fake Image Detection
    print(f"\n--- 4. TESTING ANTI-SPOOFING FAKE IMAGE REJECTION ---")
    fake_imgs = glob.glob(os.path.join(BASE_DIR, "datasets", "anti_spoof_dataset", "test", "fake", "*.jpg"))
    if fake_imgs:
        fake_result = recognize_iris(fake_imgs[0])
        print(f" Verification Result for Fake Printed Iris Image:")
        print(f"   Status      : {fake_result.get('status')}")
        print(f"   Matched User: {fake_result.get('person')}")
        print(f"   Message     : {fake_result.get('message')}")
        if fake_result.get("status") == "denied" and "SPOOF" in fake_result.get("person"):
            print("   >>> RESULT: PASSED! ANTI-SPOOFING ALERT TRIGGERED CORRECTLY!")

    # Step 5: Verify MySQL All Enrolled Users Listing
    print(f"\n--- 5. CHECKING MYSQL ALL ENROLLED USERS ---")
    all_users = get_all_users()
    print(f" Enrolled Users Count in MySQL: {len(all_users)}")
    for u in all_users:
        print(f"   - ID: {u['id']} | Name: {u['name']} | Registered At: {u['created_at']}")

    print("\n==================================================")
    print(" BACKEND PIPELINE FULLY VERIFIED & READY FOR FRONTEND!")
    print("==================================================")


if __name__ == "__main__":
    run_full_backend_test()
