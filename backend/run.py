import os
import sys
import uvicorn

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    print("==================================================")
    print(" Starting Iris Security FastAPI Backend Server...")
    print(" Local URL : http://127.0.0.1:8000")
    print(" API Docs  : http://127.0.0.1:8000/docs")
    print("==================================================")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
