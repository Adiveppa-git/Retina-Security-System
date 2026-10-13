import os
import time

now = time.time()
two_hours_ago = now - 7200
modified = []
skip_dirs = {".git", "node_modules", ".gemini", "__pycache__", "dist", "build", ".vscode", "dataset", "datasets"}

for search_dir in ["backend", "frontend/react_app/src", "ai_models", "hardware"]:
    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            p = os.path.join(root, f)
            try:
                mtime = os.path.getmtime(p)
                if mtime > two_hours_ago:
                    modified.append((p, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))))
            except Exception:
                pass

modified.sort(key=lambda x: x[1], reverse=True)
print(f"Total files modified in last 2 hours: {len(modified)}")
for p, t in modified:
    print(f"{t}: {p}")
