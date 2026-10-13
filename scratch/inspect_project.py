import os
import json
import glob

project_dir = r"c:\Users\renki\Downloads\iris-security12"

info = {}

# 1. Frontend Pages & Routes
frontend_pages_dir = os.path.join(project_dir, "frontend", "react_app", "src", "pages")
if os.path.exists(frontend_pages_dir):
    info["frontend_pages"] = [f for f in os.listdir(frontend_pages_dir) if f.endswith(('.jsx', '.js', '.tsx'))]
else:
    info["frontend_pages"] = []

# App.jsx
app_jsx_path = os.path.join(project_dir, "frontend", "react_app", "src", "App.jsx")
if os.path.exists(app_jsx_path):
    with open(app_jsx_path, "r", encoding="utf-8") as f:
        info["app_jsx_content"] = f.read()

# Navbar.jsx
navbar_path = os.path.join(project_dir, "frontend", "react_app", "src", "components", "Navbar.jsx")
if os.path.exists(navbar_path):
    with open(navbar_path, "r", encoding="utf-8") as f:
        info["navbar_content"] = f.read()

# 2. Backend Routers & Services
backend_routers_dir = os.path.join(project_dir, "backend", "app", "routers")
if os.path.exists(backend_routers_dir):
    info["backend_routers"] = [f for f in os.listdir(backend_routers_dir) if f.endswith('.py')]
else:
    info["backend_routers"] = []

backend_services_dir = os.path.join(project_dir, "backend", "app", "services")
if os.path.exists(backend_services_dir):
    info["backend_services"] = [f for f in os.listdir(backend_services_dir) if f.endswith('.py')]
else:
    info["backend_services"] = []

# 3. Backend Models & Connection
backend_models_dir = os.path.join(project_dir, "backend", "app", "models")
if os.path.exists(backend_models_dir):
    info["backend_models"] = [f for f in os.listdir(backend_models_dir) if f.endswith('.py')]

# 4. AI Models & Weights
ai_models_dir = os.path.join(project_dir, "ai_models")
if os.path.exists(ai_models_dir):
    info["ai_files"] = []
    for root, dirs, files in os.walk(ai_models_dir):
        for file in files:
            info["ai_files"].append(os.path.relpath(os.path.join(root, file), ai_models_dir))

# Print summary
print("===== PROJECT STRUCTURE INSPECTION SUMMARY =====")
print(f"Frontend Pages: {info['frontend_pages']}")
print(f"Backend Routers: {info['backend_routers']}")
print(f"Backend Services: {info['backend_services']}")
print(f"Backend Models: {info['backend_models']}")
print(f"AI Files Count: {len(info.get('ai_files', []))}")
for f in info.get("ai_files", []):
    if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".py"):
        print(f" - AI File: {f}")

with open(r"c:\Users\renki\Downloads\iris-security12\scratch\inspection_results.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
