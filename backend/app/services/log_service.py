from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
from app.models.user_model import AccessLog


def get_logs():
    """Retrieves all security access logs from MySQL database."""
    db: Session = SessionLocal()
    try:
        logs = db.query(AccessLog).order_by(AccessLog.timestamp.desc()).all()
        return [
            {
                "id": log.id,
                "Date": log.timestamp.strftime("%Y-%m-%d") if log.timestamp else "",
                "Time": log.timestamp.strftime("%H:%M:%S") if log.timestamp else "",
                "Person": log.user_name,
                "Similarity": str(log.similarity_score),
                "Status": log.status,
                "Response Time": "0.05 sec"
            }
            for log in logs
        ]
    except Exception as e:
        print("Get Logs Error:", e)
        return []
    finally:
        db.close()


def get_log_stats():
    """Computes security access statistics from MySQL logs."""
    logs = get_logs()
    total_requests = len(logs)

    if total_requests == 0:
        return {
            "total_requests": 0,
            "authorized": 0,
            "denied": 0,
            "success_rate": "0%",
            "average_response_time": "0.05 sec"
        }

    authorized = sum(1 for log in logs if log["Status"] == "ACCESS_GRANTED")
    denied = total_requests - authorized
    success_rate = round((authorized / total_requests) * 100, 2)

    return {
        "total_requests": total_requests,
        "authorized": authorized,
        "denied": denied,
        "success_rate": f"{success_rate}%",
        "average_response_time": "0.05 sec"
    }