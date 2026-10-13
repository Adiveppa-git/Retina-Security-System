from datetime import datetime, date
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
from app.models.user_model import User, AccessLog


def get_dashboard_data():
    """
    Real Database-Backed Entry/Exit Dashboard Metrics computed live from MySQL.
    """
    db: Session = SessionLocal()
    try:
        today_start = datetime.combine(date.today(), datetime.min.time())

        # 1. Total Registered Users
        total_users = db.query(User).count()

        # 2. Today's Access Logs from MySQL
        today_logs = (
            db.query(AccessLog)
            .filter(AccessLog.timestamp >= today_start)
            .order_by(AccessLog.timestamp.desc())
            .all()
        )

        checkins_today = sum(
            1 for log in today_logs
            if log.status == "ACCESS_GRANTED" and (log.action or "CHECK-IN") == "CHECK-IN"
        )
        checkouts_today = sum(
            1 for log in today_logs
            if log.status == "ACCESS_GRANTED" and log.action == "CHECK-OUT"
        )
        successful_today = sum(1 for log in today_logs if log.status == "ACCESS_GRANTED")
        denied_today = sum(1 for log in today_logs if log.status in ("ACCESS_DENIED", "SPOOF_DETECTED"))

        # 3. Determine Currently Inside Users
        # Rule: A registered user is ONLY inside if their most recent successful access log is CHECK-IN.
        all_users = db.query(User).all()
        currently_inside_list = []

        for u in all_users:
            last_granted = (
                db.query(AccessLog)
                .filter(
                    func.lower(AccessLog.user_name) == func.lower(u.name),
                    AccessLog.status == "ACCESS_GRANTED"
                )
                .order_by(AccessLog.timestamp.desc(), AccessLog.id.desc())
                .first()
            )

            if last_granted and (last_granted.action or "CHECK-IN").upper() == "CHECK-IN":
                currently_inside_list.append({
                    "user_id": u.id,
                    "user_name": u.name,
                    "checkin_time": last_granted.timestamp.strftime("%Y-%m-%d %H:%M:%S") if last_granted.timestamp else "N/A",
                    "status": "INSIDE"
                })

        currently_inside_count = len(currently_inside_list)

        # 4. Formatted Today's Access Logs
        todays_access_logs = [
            {
                "id": log.id,
                "user_name": log.user_name,
                "date": log.timestamp.strftime("%Y-%m-%d") if log.timestamp else "",
                "time": log.timestamp.strftime("%H:%M:%S") if log.timestamp else "",
                "action": log.action or "CHECK-IN",
                "status": log.status,
                "similarity_score": f"{log.similarity_score:.4f}"
            }
            for log in today_logs
        ]

        # 5. Recent Logs (Top 10 overall)
        all_logs = db.query(AccessLog).order_by(AccessLog.timestamp.desc()).limit(10).all()
        recent_logs = [
            {
                "id": log.id,
                "Date": log.timestamp.strftime("%Y-%m-%d") if log.timestamp else "",
                "Time": log.timestamp.strftime("%H:%M:%S") if log.timestamp else "",
                "Person": log.user_name,
                "Action": log.action or "CHECK-IN",
                "Status": "Success" if log.status == "ACCESS_GRANTED" else ("Fake Iris" if log.status == "SPOOF_DETECTED" else "Denied"),
                "Similarity": f"{log.similarity_score:.4f}"
            }
            for log in all_logs
        ]

        # 6. Completed Visits History (Paired Check-In & Check-Out visits)
        completed_visits = get_completed_visits_data(db, limit=20)

        return {
            "total_users": total_users,
            "currently_inside_count": currently_inside_count,
            "checkins_today": checkins_today,
            "checkouts_today": checkouts_today,
            "successful_today": successful_today,
            "denied_today": denied_today,
            "currently_inside": currently_inside_list,
            "todays_access_logs": todays_access_logs,
            "recent_logs": recent_logs,
            "completed_visits": completed_visits,
            # Backward compatibility fields
            "total_requests": len(today_logs),
            "authorized": successful_today,
            "denied": denied_today,
            "success_rate": f"{round((successful_today / len(today_logs) * 100), 1)}%" if today_logs else "0%",
            "average_response_time": "0.05 sec"
        }
    except Exception as e:
        print("Dashboard query error:", e)
        return {
            "total_users": 0,
            "currently_inside_count": 0,
            "checkins_today": 0,
            "checkouts_today": 0,
            "successful_today": 0,
            "denied_today": 0,
            "currently_inside": [],
            "todays_access_logs": [],
            "recent_logs": [],
            "completed_visits": [],
            "total_requests": 0,
            "authorized": 0,
            "denied": 0,
            "success_rate": "0%",
            "average_response_time": "0.05 sec"
        }
    finally:
        db.close()


def get_completed_visits_data(db: Session, limit: int = 20):
    """
    Pairs chronological ACCESS_GRANTED CHECK-IN and CHECK-OUT logs
    to compute exact completed visit histories and durations.
    """
    try:
        granted_logs = (
            db.query(AccessLog)
            .filter(AccessLog.status == "ACCESS_GRANTED")
            .order_by(AccessLog.timestamp.asc(), AccessLog.id.asc())
            .all()
        )
        open_visits = {}
        completed = []
        for log in granted_logs:
            uname = (log.user_name or "").strip()
            uname_key = uname.lower()
            act = (log.action or "CHECK-IN").upper()
            if act == "CHECK-IN":
                open_visits[uname_key] = log
            elif act == "CHECK-OUT" and uname_key in open_visits:
                in_log = open_visits.pop(uname_key)
                if in_log.timestamp and log.timestamp:
                    diff_sec = max(0, int((log.timestamp - in_log.timestamp).total_seconds()))
                    hrs = diff_sec // 3600
                    mins = (diff_sec % 3600) // 60
                    secs = diff_sec % 60
                    dur_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"
                else:
                    dur_str = "00h 00m 00s"
                completed.append({
                    "user_name": uname,
                    "check_in_time": in_log.timestamp.isoformat() if in_log.timestamp else "N/A",
                    "check_in_formatted": in_log.timestamp.strftime("%I:%M:%S %p") if in_log.timestamp else "N/A",
                    "check_out_time": log.timestamp.isoformat() if log.timestamp else "N/A",
                    "check_out_formatted": log.timestamp.strftime("%I:%M:%S %p") if log.timestamp else "N/A",
                    "duration": dur_str
                })
        completed.reverse()
        return completed[:limit]
    except Exception as e:
        print("Error getting completed visits:", e)
        return []