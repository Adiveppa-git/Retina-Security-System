from fastapi import APIRouter
from backend.app.services.log_service import get_logs, get_log_stats

router = APIRouter(
    prefix="/logs",
    tags=["Logs"]
)


@router.get("/")
def read_logs():
    return get_logs()


@router.get("/stats")
def read_log_stats():
    return get_log_stats()