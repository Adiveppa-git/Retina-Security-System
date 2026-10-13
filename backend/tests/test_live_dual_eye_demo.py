import requests
import os
import glob

API_URL = "http://127.0.0.1:8000"


def test_dual_eye_verification():
    print("==================================================")
    print(" TESTING DUAL-EYE BIOMETRIC VERIFICATION PIPELINE")
    print("==================================================")

    test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
    subjs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))]

    if len(subjs) < 2:
        return

    subj1_path = os.path.join(test_rec_dir, subjs[0])
    subj2_path = os.path.join(test_rec_dir, subjs[1])

    imgs1 = glob.glob(os.path.join(subj1_path, "*.jpg"))
    imgs2 = glob.glob(os.path.join(subj2_path, "*.jpg"))

    name = f"Registered_Friend_{subjs[0]}"

    # Step 1: Register User
    with open(imgs1[0], "rb") as f:
        files = [("files", (os.path.basename(imgs1[0]), f, "image/jpeg"))]
        data = {"name": name, "email": "friend@example.com"}
        reg_res = requests.post(f"{API_URL}/user/add", data=data, files=files)

    print(f" 1. Registration Output [{name}]: {reg_res.json()}")

    # Step 2: Verify Registered User (SHOULD GRANT ACCESS)
    with open(imgs1[1], "rb") as f:
        files = {"file": (os.path.basename(imgs1[1]), f, "image/jpeg")}
        v1_res = requests.post(f"{API_URL}/iris/verify", files=files)

    v1_json = v1_res.json()
    print(f"\n 2. Verification Result for Registered User '{name}':")
    print(f"   Status      : {v1_json.get('status')}")
    print(f"   Matched User: {v1_json.get('person')}")
    print(f"   Similarity  : {v1_json.get('similarity')}%")
    print(f"   Message     : {v1_json.get('message')}")
    if v1_json.get('status') == 'success':
        print("   >>> VERIFICATION RESULT: PASSED! REGISTERED FRIEND GRANTED ACCESS!")

    # Step 3: Verify Unregistered User (SHOULD DENY ACCESS)
    with open(imgs2[0], "rb") as f:
        files = {"file": (os.path.basename(imgs2[0]), f, "image/jpeg")}
        v2_res = requests.post(f"{API_URL}/iris/verify", files=files)

    v2_json = v2_res.json()
    print(f"\n 3. Verification Result for Unregistered User:")
    print(f"   Status      : {v2_json.get('status')}")
    print(f"   Matched User: {v2_json.get('person')}")
    print(f"   Similarity  : {v2_json.get('similarity')}%")
    print(f"   Message     : {v2_json.get('message')}")
    if v2_json.get('status') == 'denied':
        print("   >>> VERIFICATION RESULT: PASSED! UNREGISTERED STRANGER DENIED ACCESS!")

    print("\n==================================================")
    print(" DUAL-EYE BIOMETRIC PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_dual_eye_verification()
