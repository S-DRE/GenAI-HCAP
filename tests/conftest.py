"""tests/conftest.py — runs before any test module is imported.

1. Loads the .env file so that environment variables (GROQ_API_KEY, etc.)
   are available during collection, including pytest.mark.skipif checks
   evaluated at module import time (e.g. in test_e2e.py).

2. Provides a session-scoped fixture that verifies the live server is
   reachable AND not returning errors before any E2E test runs.
"""

import os

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session", autouse=False)
def require_live_server():
    """Skip the entire E2E session if the server is unreachable or unhealthy.

    Attach this fixture to any test class or function that needs a live server.
    It runs once per session and fails fast with a clear message rather than
    letting every test fail with a cryptic connection/JSON error.
    """
    base_url = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
    try:
        r = httpx.get(f"{base_url}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Live server not reachable at {base_url} — {exc}. Start with: uvicorn src.api.main:app")
