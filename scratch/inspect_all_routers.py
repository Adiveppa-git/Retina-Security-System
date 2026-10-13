import os

routers_dir = r"c:\Users\renki\Downloads\iris-security12\backend\app\routers"
for fname in os.listdir(routers_dir):
    if fname.endswith(".py") and fname != "__init__.py":
        path = os.path.join(routers_dir, fname)
        print(f"\n================ {fname} ================")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("@router."):
                    print(f"Line {i+1}: {line.strip()}")
                    if i + 1 < len(lines):
                        print(f"          {lines[i+1].strip()}")
