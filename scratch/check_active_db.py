import sys, os
sys.path.insert(0, os.path.abspath("backend"))
from app.database.connection import engine
from sqlalchemy import inspect, text

inspector = inspect(engine)
print("Engine url:", engine.url)
print("Table names:", inspector.get_table_names())

with engine.connect() as conn:
    for t in inspector.get_table_names():
        count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"Table {t}: {count} rows")
        if count > 0:
            sample = conn.execute(text(f"SELECT * FROM {t} LIMIT 3")).fetchall()
            print("  Sample:", sample)
