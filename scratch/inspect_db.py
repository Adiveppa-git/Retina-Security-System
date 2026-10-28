import sqlite3

conn = sqlite3.connect("iris_security_db.sqlite3")
cursor = conn.cursor()

tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)

for (table_name,) in tables:
    cols = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    print(f"\nTable: {table_name}")
    for col in cols:
        print(f"  {col[1]} ({col[2]})")
    rows = cursor.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()
    print(f"  Sample rows ({len(rows)}):")
    for r in rows:
        # print without huge embeddings
        print("   ", [str(x)[:50] if isinstance(x, str) else x for x in r])

conn.close()
