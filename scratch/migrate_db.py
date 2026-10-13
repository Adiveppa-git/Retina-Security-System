import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.database.connection import SessionLocal
from sqlalchemy import text, inspect

db = SessionLocal()
try:
    engine = db.get_bind()
    inspector = inspect(engine)
    cols = [c['name'] for c in inspector.get_columns('access_logs')]

    if 'action' not in cols:
        print("Adding 'action' column to 'access_logs' table...")
        db.execute(text("ALTER TABLE access_logs ADD COLUMN action VARCHAR(50) DEFAULT 'CHECK-IN'"))
        db.commit()
        print("Migration Success: 'action' column added to access_logs table.")
    else:
        print("Column 'action' already exists in 'access_logs' table.")
except Exception as e:
    print("Migration warning:", e)
finally:
    db.close()
