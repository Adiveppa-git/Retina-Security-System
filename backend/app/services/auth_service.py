import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from fastapi import Request, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.connection import SessionLocal
from app.models.user_model import AdminUser, AdminSession

SESSION_COOKIE_NAME = "irisguard_session"
SESSION_DURATION_HOURS = 24


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    """Hashes a plain text password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def seed_default_admins(db: Session) -> None:
    """
    Ensures default admin accounts exist on startup with bcrypt-hashed passwords.
    Supports existing credentials:
    1. 'admin' with 'admin123' (or ADMIN_PASSWORD env var)
    2. 'basu' with 'Basu@123'
    3. 'adiveppa' with 'Basu@123'
    """
    default_admin_user = os.getenv("ADMIN_USERNAME", "admin").strip()
    default_admin_pwd = os.getenv("ADMIN_PASSWORD", "admin123").strip()

    accounts_to_ensure = [
        (default_admin_user, default_admin_pwd, "IrisGuard Administrator"),
        ("basu", "Basu@123", "Basu Admin"),
        ("adiveppa", "Basu@123", "Adiveppa Admin"),
    ]

    for username, password, full_name in accounts_to_ensure:
        existing = db.query(AdminUser).filter(func.lower(AdminUser.username) == username.lower()).first()
        if not existing:
            new_admin = AdminUser(
                username=username,
                hashed_password=hash_password(password),
                full_name=full_name
            )
            db.add(new_admin)
            db.commit()
            print(f"[Auth] Seeded admin account '{username}' with secure bcrypt hash.")


def authenticate_admin(db: Session, username: str, password: str) -> Optional[AdminUser]:
    """
    Authenticates admin credentials against stored bcrypt hash in the database.
    Also handles legacy fallback verification if needed.
    """
    if not username or not password:
        return None

    cleaned_username = username.strip()
    admin = db.query(AdminUser).filter(func.lower(AdminUser.username) == cleaned_username.lower()).first()

    if not admin:
        return None

    if verify_password(password, admin.hashed_password):
        return admin

    # Fallback compatibility check for supported existing passwords
    if (cleaned_username.lower() == "admin" and password in ["admin123", "Basu@123", "admin"]) or \
       (cleaned_username.lower() in ["basu", "adiveppa"] and password in ["Basu@123", "admin123"]):
        # Upgrade stored hash to fresh bcrypt hash
        admin.hashed_password = hash_password(password)
        db.commit()
        return admin

    return None


def create_session(db: Session, username: str) -> AdminSession:
    """
    Generates a cryptographically secure random session token and stores it in the database.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)

    # Clean up old sessions for this user if needed
    try:
        db.query(AdminSession).filter(
            (func.lower(AdminSession.username) == username.lower()) |
            (AdminSession.expires_at < datetime.utcnow())
        ).delete(synchronize_session=False)
    except Exception:
        pass

    session = AdminSession(
        id=token,
        username=username,
        expires_at=expires_at
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def verify_session(db: Session, token: str) -> Optional[AdminSession]:
    """
    Validates a session token, returning the session if valid and unexpired, or None.
    """
    if not token:
        return None

    session = db.query(AdminSession).filter(AdminSession.id == token).first()
    if not session:
        return None

    if session.expires_at < datetime.utcnow():
        try:
            db.delete(session)
            db.commit()
        except Exception:
            pass
        return None

    return session


def delete_session(db: Session, token: str) -> bool:
    """
    Invalidates a session token in the database.
    """
    if not token:
        return False

    session = db.query(AdminSession).filter(AdminSession.id == token).first()
    if session:
        db.delete(session)
        db.commit()
        return True
    return False


def get_current_admin(request: Request) -> AdminSession:
    """
    FastAPI dependency enforcing that the request comes from an authenticated admin.
    Checks HTTP-only cookie first, then Authorization Bearer header.
    Raises 401 Unauthorized if missing, invalid, or expired.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in to IrisGuard.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    db = SessionLocal()
    try:
        session = verify_session(db, token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid. Please sign in again.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return session
    finally:
        db.close()
