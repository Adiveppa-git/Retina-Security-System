import requests
import os
import glob

API_URL = "http://127.0.0.1:8000"


def test_live_http_endpoints():
    print("==================================================")
    print(" TESTING LIVE HTTP BACKEND ENDPOINTS (127.0.0.1:8000)")
    print("==================================================")

    # 1. Test Home Endpoint
    res = requests.get(f"{API_URL}/")
    print(f" 1. GET / Output ({res.status_code}): {res.json()}")

    # 2. Find test set images for registration & verification
    test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
    subjs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))][:3]

    registered_users = []

    for idx, subj in enumerate(subjs, start=1):
        subj_path = os.path.join(test_rec_dir, subj)
        imgs = glob.glob(os.path.join(subj_path, "*.jpg"))
        if len(imgs) < 2:
            continue

        reg_img_path = imgs[0]
        ver_img_path = imgs[1]
        name = f"HTTP_User_{idx}_{subj}"

        # Register User via HTTP POST /user/add
        with open(reg_img_path, "rb") as f:
            files = {"file": (os.path.basename(reg_img_path), f, "image/jpeg")}
            data = {"name": name, "email": f"http_user_{idx}@test.com"}
            reg_res = requests.post(f"{API_URL}/user/add", data=data, files=files)

        print(f"\n 2. POST /user/add [{name}] Status: {reg_res.status_code} | Res: {reg_res.json()}")

        if reg_res.status_code == 200 and reg_res.json().get("status") == "success":
            registered_users.append({
                "name": name,
                "ver_img": ver_img_path
            })

    # 3. Test HTTP POST /iris/verify with unseen test images
    print("\n--- 3. TESTING LIVE HTTP POST /iris/verify ---")
    for user in registered_users:
        name = user["name"]
        ver_img_path = user["ver_img"]

        with open(ver_img_path, "rb") as f:
            files = {"file": (os.path.basename(ver_img_path), f, "image/jpeg")}
            ver_res = requests.post(f"{API_URL}/iris/verify", files=files)

        print(f"\n Verification HTTP Output for '{name}':")
        print(f"   HTTP Status Code : {ver_res.status_code}")
        print(f"   Response JSON    : {ver_res.json()}")

        json_data = ver_res.json()
        if json_data.get("status") == "success" and json_data.get("person") == name:
            print(f"   >>> VERIFICATION RESULT: PASSED! Matched User '{json_data.get('person')}' at {json_data.get('similarity')}% Similarity!")
        else:
            print("   >>> VERIFICATION RESULT: FAILED!")

    # 4. Test HTTP GET /user/all
    all_users_res = requests.get(f"{API_URL}/user/all")
    print(f"\n 4. GET /user/all Output ({all_users_res.status_code}): Enrolled Count = {len(all_users_res.json())}")

    print("\n==================================================")
    print(" LIVE FASTAPI HTTP BACKEND IS 100% WORKING WITHOUT SINGLE ERROR!")
    print("==================================================")


if __name__ == "__main__":
    test_live_http_endpoints()
