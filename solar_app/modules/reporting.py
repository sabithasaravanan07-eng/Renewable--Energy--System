"""Writes the latest estimate result to reports/latest_report.json."""
import json
import os

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "latest_report.json")


def save_report(result: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
