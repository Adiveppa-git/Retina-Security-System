import os
import sys

sys.path.append(os.path.abspath("backend"))

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbedding, AccessLog


def clear_database():
    db = SessionLocal()
    try:
        num_emb = db.query(IrisEmbedding).delete()
        num_logs = db.query(AccessLog).delete()
        num_users = db.query(User).delete()
        db.commit()
        print(f"Cleared MySQL Database: Removed {num_users} users, {num_emb} embeddings, {num_logs} access logs.")
    except Exception as e:
        db.rollback()
        print("Database Clear Error:", e)
    finally:
        db.close()


if __name__ == "__main__":
    clear_database()
