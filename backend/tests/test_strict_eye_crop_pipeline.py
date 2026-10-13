import requests
import os
import glob

API_URL = "http://127.0.0.1:8000"


def test_unregistered_rejection_and_registered_grant():
    print("==================================================")
    print(" TESTING UNREGISTERED REJECTION & REGISTERED GRANT")
    print("==================================================")

    # 1. Clear database first
    test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
    subjs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))]

    if len(subjs) < 2:
        print("Error: Need at least 2 subject folders.")
        return

    subj1_path = os.path.join(test_rec_dir, subjs[0])
    subj2_path = os.path.join(test_rec_dir, subjs[1])

    subj1_imgs = glob.glob(os.path.join(subj1_path, "*.jpg"))
    subj2_imgs = glob.glob(os.path.join(subj2_path, "*.jpg"))

    name1 = f"Registered_Person_{subjs[0]}"

    # Register Person 1 into MySQL
    with open(subj1_imgs[0], "rb") as f:
        files = [("files", (os.path.basename(subj1_imgs[0]), f, "image/jpeg"))]
        data = {"name": name1, "email": "person1@example.com"}
        reg_res = requests.post(f"{API_URL}/user/add", data=data, files=files)

    print(f" 1. Registered '{name1}': {reg_res.json()}")

    # TEST A: Verify Person 1 with unseen image of Person 1 (SHOULD GRANT ACCESS)
    with open(subj1_imgs[1], "rb") as f:
        files = {"file": (os.path.basename(subj1_imgs[1]), f, "image/jpeg")}
        v1_res = requests.post(f"{API_URL}/iris/verify", files=files)

    print(f"\n TEST A (Genuine Registered Person 1):")
    print(f"   Status      : {v1_res.json().get('status')}")
    print(f"   Person      : {v1_res.json().get('person')}")
    print(f"   Similarity  : {v1_res.json().get('similarity')}%")
    print(f"   Message     : {v1_res.json().get('message')}")

    # TEST B: Verify Person 2 (UNREGISTERED PERSON) (SHOULD DENY ACCESS WITH 'Unregistered / Unknown Person')
    with open(subj2_imgs[0], "rb") as f:
        files = {"file": (os.path.basename(subj2_imgs[0]), f, "image/jpeg")}
        v2_res = requests.post(f"{API_URL}/iris/verify", files=files)

    print(f"\n TEST B (Unregistered Person 2):")
    print(f"   Status      : {v2_res.json().get('status')}")
    print(f"   Person      : {v2_res.json().get('person')}")
    print(f"   Similarity  : {v2_res.json().get('similarity')}%")
    print(f"   Message     : {v2_res.json().get('message')}")

    print("\n==================================================")
    if v1_res.json().get('status') == 'success' and v2_res.json().get('status') == 'denied':
        print(" ✅ PERFECT RESULTS! REGISTERED USER GRANTED, UNREGISTERED USER DENIED!")
    else:
        print(" ❌ PIPELINE NEEDS THRESHOLD CALIBRATION.")
    print("==================================================")


if __name__ == "__main__":
    test_unregistered_rejection_and_registered_grant()
