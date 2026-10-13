import os

models_file = r"c:\Users\renki\Downloads\iris-security12\backend\app\models\user_model.py"
iris_router_file = r"c:\Users\renki\Downloads\iris-security12\backend\app\routers\iris_router.py"

print("--- USER MODEL CONTENT ---")
if os.path.exists(models_file):
    with open(models_file, "r", encoding="utf-8") as f:
        print(f.read())

print("--- IRIS ROUTER ENDPOINTS ---")
if os.path.exists(iris_router_file):
    with open(iris_router_file, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()
        for line in lines:
            if line.strip().startswith("@router."):
                print(line.strip())
