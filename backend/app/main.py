import os
import sys

# Ensure workspace root directory is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.connection import engine, Base
from app.routers import (
    iris_router,
    user_router,
    log_router,
    dashboard_router
)

# Auto-create MySQL Tables on Startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Iris Security Biometric Backend API", openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(iris_router.router)
app.include_router(user_router.router)
app.include_router(log_router.router)
app.include_router(dashboard_router.router)


@app.get("/")
def home():
    return {
        "status": "online",
        "system": "Iris Security System Backend API",
        "version": "1.0.0"
    }