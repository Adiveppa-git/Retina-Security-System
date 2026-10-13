import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.database.connection import engine
from sqlalchemy import text, inspect

try:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("--- TABLES IN DATABASE ---")
    print(tables)

    with engine.connect() as conn:
        for t in tables:
            cols = inspector.get_columns(t)
            print(f"\nTable: '{t}'")
            for c in cols:
                print(f"  - {c['name']} | Type: {c['type']} | Nullable: {c['nullable']} | Default: {c['default']}")
            count = conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar()
            print(f"  Total Rows: {count}")

    print("\nDatabase dialect:", engine.name)
    print("Database URL:", engine.url)
except Exception as e:
    print("Error during inspection:", e)
