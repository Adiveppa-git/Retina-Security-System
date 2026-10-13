import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.database.connection import SessionLocal
from app.models.user_model import User, AccessLog
from app.services.iris_recognition_service import log_access_event
import requests

BASE_URL = "http://127.0.0.1:8000"

db = SessionLocal()
u = db.query(User).first()
print(f"Simulating ACCESS_DENIED attempt for '{u.name}' during CHECK-IN...")
log_access_event(db, u.name, "ACCESS_DENIED", 0.5120, action="CHECK-IN")
db.close()

# Fetch dashboard data from FastAPI endpoint
res = requests.get(f"{BASE_URL}/dashboard/").json()
print("\nPost-Denied Dashboard Data:")
print("  - Total Users:", res["total_users"])
print("  - Currently Inside Count:", res["currently_inside_count"])
print("  - Check-ins Today:", res["checkins_today"])
print("  - Check-outs Today:", res["checkouts_today"])
print("  - Denied Today:", res["denied_today"])
print("  - Currently Inside Table:", res["currently_inside"])
