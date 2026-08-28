import os
import torch

from backend.app.services.log_service import get_logs, get_log_stats

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

USER_DB = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/user_features.pth"
)


def get_dashboard_data():

    # ----------------------------
    # Total Registered Users
    # ----------------------------
    total_users = 0

    if os.path.exists(USER_DB):

        try:
            db = torch.load(USER_DB)
            total_users = len(db["names"])

        except Exception:
            total_users = 0

    # ----------------------------
    # Logs
    # ----------------------------
    logs = get_logs()

    stats = get_log_stats()

    # ----------------------------
    # Recent Logs
    # ----------------------------
    recent_logs = logs[-5:]

    recent_logs.reverse()

    # ----------------------------
    # Dashboard Response
    # ----------------------------
    return {

        "total_users": total_users,

        "total_requests": stats["total_requests"],

        "authorized": stats["authorized"],

        "denied": stats["denied"],

        "success_rate": stats["success_rate"],

        "average_response_time": stats["average_response_time"],

        "recent_logs": recent_logs

    }