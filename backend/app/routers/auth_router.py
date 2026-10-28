from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from app.services.auth_service import (
    get_db,
    authenticate_admin,
    create_session,
    verify_session,
    delete_session,
    get_current_admin,
    SESSION_COOKIE_NAME,
    SESSION_DURATION_HOURS
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(login_data: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """
    Authenticates the admin user and sets an HTTP-only session cookie.
    Also returns the token in JSON for clients using Bearer tokens.
    """
    username = login_data.username.strip() if login_data.username else ""
    password = login_data.password.strip() if login_data.password else ""

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required."
        )

    admin = authenticate_admin(db, username=username, password=password)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid administrator credentials. Please check your username and password."
        )

    session = create_session(db, username=admin.username)

    # Set secure HTTP-only cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.id,
        httponly=True,
        samesite="lax",
        secure=False,  # Suitable for local HTTP development; can be True in HTTPS production
        max_age=SESSION_DURATION_HOURS * 3600,
        path="/"
    )

    return {
        "status": "success",
        "message": "Authentication successful.",
        "token": session.id,
        "username": admin.username,
        "full_name": admin.full_name
    }


@router.get("/me")
def check_session(request: Request, current_admin=Depends(get_current_admin)):
    """
    Validates current admin session status.
    Returns admin details if session is active; raises 401 otherwise.
    """
    return {
        "authenticated": True,
        "username": current_admin.username,
        "session_id": current_admin.id
    }


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """
    Invalidates current admin session in the database and clears the session cookie.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if token:
        delete_session(db, token)

    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")

    return {
        "status": "success",
        "message": "Signed out successfully."
    }
