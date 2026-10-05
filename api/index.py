"""Vercel serverless entry point for the unified Flask dashboard."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
