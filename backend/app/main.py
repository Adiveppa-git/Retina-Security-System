from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import (
    iris_router,
    user_router,
    log_router,
    dashboard_router
)

app = FastAPI(openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(iris_router.router)
app.include_router(user_router.router)
app.include_router(log_router.router)
app.include_router(dashboard_router.router)


@app.get("/")
def home():
    return {
        "message": "Iris Security Backend Running"
    }