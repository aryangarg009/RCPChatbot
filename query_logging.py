from __future__ import annotations

import csv
import threading
from pathlib import Path

from config import ENABLE_QUERY_LOG_CSV, QUERY_LOG_CSV_PATH

_LOG_HEADERS = ["input_query", "latency_ms", "path", "type", "output"]
_WRITE_LOCK = threading.Lock()
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY_LOG_CSV_PATH = _PROJECT_ROOT / "query_metrics_log.csv"
PROTECTED_QUERY_LOG_PATHS = {
    (_PROJECT_ROOT / "query_metrics_log_evaluation_run.csv").resolve(),
}


def _resolve_log_path() -> Path:
    path = Path(QUERY_LOG_CSV_PATH)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    resolved = path.resolve()
    if resolved in PROTECTED_QUERY_LOG_PATHS:
        return DEFAULT_QUERY_LOG_CSV_PATH
    return resolved


def log_query_row(
    input_query: str,
    latency_ms: float,
    execution_path: str,
    response_type: str,
    output: str,
) -> None:
    """
    Append one query execution row to CSV.
    Logging failures are intentionally swallowed by callers to avoid blocking chat responses.
    """
    if not ENABLE_QUERY_LOG_CSV:
        return

    csv_path = _resolve_log_path()
    row = [input_query, f"{latency_ms:.2f}", execution_path, response_type, output]

    with _WRITE_LOCK:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_LOG_HEADERS)
            writer.writerow(row)
