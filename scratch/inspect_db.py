import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.database.connection import SessionLocal
from sqlalchemy import inspect

try:
    engine = SessionLocal().get_bind()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("MySQL Tables Discovered:", tables)

    for t in tables:
        cols = inspector.get_columns(t)
        print(f"\nTable '{t}' Columns:")
        for c in cols:
            print(f"  - {c['name']} ({c['type']})")
except Exception as e:
    print("Database inspection error:", e)
