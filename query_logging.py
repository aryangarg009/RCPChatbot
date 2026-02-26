from __future__ import annotations

import csv
import threading
from pathlib import Path

from config import ENABLE_QUERY_LOG_CSV, QUERY_LOG_CSV_PATH

_LOG_HEADERS = ["input_query", "latency_ms", "path", "type", "output"]
_WRITE_LOCK = threading.Lock()


def _resolve_log_path() -> Path:
    path = Path(QUERY_LOG_CSV_PATH)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


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
