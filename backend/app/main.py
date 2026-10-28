import os
import sys

# Ensure workspace root directory is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

import app.models.user_model  # Register all models with Base.metadata
from app.database.connection import engine, Base, SessionLocal
from app.routers import (
    iris_router,
    user_router,
    log_router,
    dashboard_router,
    auth_router
)
from app.services.auth_service import get_current_admin, seed_default_admins

# Auto-create MySQL Tables on Startup
Base.metadata.create_all(bind=engine)

# Auto-seed default admins with secure bcrypt hashes
try:
    with SessionLocal() as db:
        seed_default_admins(db)
except Exception as e:
    print(f"[Auth Warning] Failed to seed default admin: {e}")

app = FastAPI(title="IrisGuard Biometric Backend API", openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication router (public endpoints: /auth/login, /auth/logout, /auth/me)
app.include_router(auth_router.router)

# Protected API Routers (strictly enforce authentication on all operations)
app.include_router(iris_router.router, dependencies=[Depends(get_current_admin)])
app.include_router(user_router.router, dependencies=[Depends(get_current_admin)])
app.include_router(log_router.router, dependencies=[Depends(get_current_admin)])
app.include_router(dashboard_router.router, dependencies=[Depends(get_current_admin)])


@app.get("/")
def home():
    return {
        "status": "online",
        "system": "IrisGuard Backend API",
        "version": "1.0.0"
    }