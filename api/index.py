"""Vercel serverless entry point for the unified Flask dashboard."""
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlencode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402


class VercelPathMiddleware:
    """Restore the requested Flask route after Vercel rewrites to one function."""

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        original_paths = query.pop("__flask_path", [])
        if original_paths:
            original_path = original_paths[0]
            if original_path.startswith("/") and not original_path.startswith("//"):
                environ["PATH_INFO"] = original_path
            environ["QUERY_STRING"] = urlencode(query, doseq=True)
        return self.application(environ, start_response)


app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
