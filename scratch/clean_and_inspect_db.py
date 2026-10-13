import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.database.connection import SessionLocal
from sqlalchemy import text, inspect

db = SessionLocal()
try:
    # 1. Clean test records inserted during scratch testing if any
    db.execute(text("DELETE FROM access_logs WHERE user_name='Adiveppa' AND similarity_score IN (0.9421, 0.9315, 0.5120)"))
    db.commit()
    print("Cleaned scratch test logs.")

    # 2. Inspect MySQL tables and columns
    engine = db.get_bind()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("\n--- MYSQL TABLES DISCOVERED ---")
    print(tables)

    for t in tables:
        cols = inspector.get_columns(t)
        print(f"\nTable: '{t}'")
        for c in cols:
            print(f"  - {c['name']} | Type: {c['type']} | Nullable: {c['nullable']} | Default: {c['default']}")

        # Count rows
        count = db.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"  Total Rows: {count}")

        # Show sample row if exists
        sample = db.execute(text(f"SELECT * FROM {t} LIMIT 1")).fetchone()
        if sample:
            print(f"  Sample Row: {sample}")

except Exception as e:
    print("Error:", e)
finally:
    db.close()
