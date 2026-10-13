import os
import socket
import urllib.parse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.database.base import Base

password = urllib.parse.quote_plus("Basu@123")
DEFAULT_MYSQL_URL = f"mysql+pymysql://root:{password}@127.0.0.1:3306/iris_security_db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_MYSQL_URL)

def can_reach_mysql(host="127.0.0.1", port=3306, timeout=0.8):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def create_db_engine():
    if DATABASE_URL.startswith("sqlite"):
        return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

    if "127.0.0.1" in DATABASE_URL or "localhost" in DATABASE_URL:
        if not can_reach_mysql("127.0.0.1", 3306, timeout=0.8):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            sqlite_file = os.path.join(base_dir, "iris_security_db.sqlite3")
            sqlite_url = f"sqlite:///{sqlite_file}"
            print(f"[Database] MySQL not running on localhost:3306. Using local SQLite: {sqlite_url}")
            return create_engine(sqlite_url, connect_args={"check_same_thread": False})

    try:
        test_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 2}
        )
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[Database] Successfully connected to MySQL database.")
        return test_engine
    except Exception as e:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        sqlite_file = os.path.join(base_dir, "iris_security_db.sqlite3")
        sqlite_url = f"sqlite:///{sqlite_file}"
        print(f"[Database] MySQL connection failed ({e}).")
        print(f"[Database] Seamlessly falling back to local SQLite database: {sqlite_url}")
        return create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False}
        )

engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
