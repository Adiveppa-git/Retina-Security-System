import csv
import os
from datetime import datetime


# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "access_log.csv"
)


# ==========================================
# Save Access Log
# ==========================================

def save_log(person, similarity, status, response_time):

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Time",
                "Person",
                "Similarity",
                "Status",
                "Response Time"
            ])

        now = datetime.now()

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            person,
            similarity,
            status,
            response_time
        ])


# ==========================================
# Get All Logs
# ==========================================

def get_logs():

    if not os.path.exists(LOG_FILE):
        return []

    logs = []

    with open(LOG_FILE, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:
            logs.append(row)

    return logs


# ==========================================
# Log Statistics
# ==========================================

def get_log_stats():

    logs = get_logs()

    total_requests = len(logs)

    if total_requests == 0:
        return {
            "total_requests": 0,
            "authorized": 0,
            "denied": 0,
            "success_rate": "0%",
            "average_response_time": "0 sec"
        }

    authorized = sum(
        1 for log in logs
        if log["Status"] == "Success"
    )

    denied = total_requests - authorized

    success_rate = round(
        (authorized / total_requests) * 100,
        2
    )

    total_time = 0

    for log in logs:

        try:
            total_time += float(
                log["Response Time"].replace(" sec", "")
            )
        except:
            pass

    average_time = round(
        total_time / total_requests,
        3
    )

    return {

        "total_requests": total_requests,

        "authorized": authorized,

        "denied": denied,

        "success_rate": f"{success_rate}%",

        "average_response_time": f"{average_time} sec"

    }