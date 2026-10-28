import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
backend_root = os.path.join(ROOT, "backend")
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.main import app

from app.services.auth_service import get_current_admin
from app.models.user_model import AdminSession


@pytest.fixture(autouse=True)
def auto_authenticate_tests():
    """
    FastAPI dependency override for existing biometric/hardware test suites,
    allowing existing functional tests to run with simulated admin permissions.
    """
    mock_session = AdminSession(
        id="test-admin-session-token",
        username="admin"
    )
    app.dependency_overrides[get_current_admin] = lambda: mock_session
    yield
    app.dependency_overrides.pop(get_current_admin, None)
