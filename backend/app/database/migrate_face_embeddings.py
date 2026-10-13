"""
Database Migration: Face Embeddings Table
=========================================
Standard SQLAlchemy schema migration to ensure 'face_embeddings' table
is created with proper Foreign Key to 'users.id' (ON DELETE CASCADE)
without touching any existing tables or rows.
"""

import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database.connection import engine
from app.models.user_model import Base, FaceEmbedding
from sqlalchemy import inspect


def migrate_face_embeddings_table():
    """
    Creates face_embeddings table using standard SQLAlchemy DDL with checkfirst=True.
    Leaves all other tables and data intact.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print("[Migration] Database connection established.")
    print(f"[Migration] Existing tables before migration: {existing_tables}")

    if "face_embeddings" in existing_tables:
        print("[Migration] Table 'face_embeddings' already exists.")
    else:
        print("[Migration] Creating 'face_embeddings' table...")
        FaceEmbedding.__table__.create(bind=engine, checkfirst=True)
        print("[Migration] Table 'face_embeddings' created successfully.")

    # Verification
    inspector = inspect(engine)
    updated_tables = inspector.get_table_names()
    print(f"[Migration] Updated tables in database: {updated_tables}")

    if "face_embeddings" in updated_tables:
        print("\n--- 'face_embeddings' SCHEMA VERIFICATION ---")
        cols = inspector.get_columns("face_embeddings")
        for c in cols:
            print(f"  - Column: {c['name']} | Type: {c['type']} | Nullable: {c['nullable']} | Default: {c['default']}")

        fks = inspector.get_foreign_keys("face_embeddings")
        print("\n--- FOREIGN KEYS ---")
        for fk in fks:
            print(f"  - FK: {fk['name']} | Constrained: {fk['constrained_columns']} -> Referred: {fk['referred_table']}.{fk['referred_columns']}")

        pk = inspector.get_pk_constraint("face_embeddings")
        print(f"\n--- PRIMARY KEY: {pk['constrained_columns']} ---")
        return True
    else:
        print("[Migration ERROR] 'face_embeddings' table was not found after creation.")
        return False


if __name__ == "__main__":
    success = migrate_face_embeddings_table()
    if not success:
        sys.exit(1)
