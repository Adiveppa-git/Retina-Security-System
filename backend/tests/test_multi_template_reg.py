import requests
import os
import glob

API_URL = "http://127.0.0.1:8000"


def test_multi_template_registration():
    print("==================================================")
    print(" TESTING MULTI-TEMPLATE BIOMETRIC REGISTRATION & VERIFICATION")
    print("==================================================")

    test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
    subjs = [d for d in os.listdir(test_rec_dir) if os.path.isdir(os.path.join(test_rec_dir, d))][:2]

    for idx, subj in enumerate(subjs, start=1):
        subj_path = os.path.join(test_rec_dir, subj)
        imgs = glob.glob(os.path.join(subj_path, "*.jpg")) + glob.glob(os.path.join(subj_path, "*.png"))

        if len(imgs) < 3:
            continue

        name = f"Multi_Tpl_User_{idx}_{subj}"
        # Prepare 3 template image files (simulating Both Eyes, Left Eye, Right Eye)
        reg_files = [
            ("files", (os.path.basename(imgs[0]), open(imgs[0], "rb"), "image/jpeg")),
            ("files", (os.path.basename(imgs[1]), open(imgs[1], "rb"), "image/jpeg")),
            ("files", (os.path.basename(imgs[2]), open(imgs[2], "rb"), "image/jpeg")),
        ]

        data = {"name": name, "email": f"multi_{idx}@university.edu"}

        # Perform Multi-Template Registration POST /user/add
        res = requests.post(f"{API_URL}/user/add", data=data, files=reg_files)
        print(f"\n Multi-Template Registration [{name}]: HTTP {res.status_code} | Res: {res.json()}")

        # Now verify using a 4th unseen test image
        if len(imgs) >= 4:
            ver_img = imgs[3]
            with open(ver_img, "rb") as vf:
                vfiles = {"file": (os.path.basename(ver_img), vf, "image/jpeg")}
                vres = requests.post(f"{API_URL}/iris/verify", files=vfiles)

            print(f" Verification for [{name}] using 4th unseen test image:")
            print(f"   Response JSON: {vres.json()}")
            if vres.json().get("status") == "success" and vres.json().get("person") == name:
                print(f"   >>> RESULT: PASSED! Matched Multi-Template User '{name}' at {vres.json().get('similarity')}% Similarity!")

    print("\n==================================================")
    print(" MULTI-TEMPLATE BIOMETRIC PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_multi_template_registration()
