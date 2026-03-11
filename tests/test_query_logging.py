import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import query_logging


def test_evaluation_log_csv_is_protected() -> None:
    protected = (Path(query_logging.__file__).resolve().parent / "query_metrics_log_evaluation_run.csv").resolve()
    assert protected in query_logging.PROTECTED_QUERY_LOG_PATHS


def test_log_query_row_redirects_protected_path(monkeypatch, tmp_path) -> None:
    protected = tmp_path / "query_metrics_log_evaluation_run.csv"
    protected.write_text("keep as fixture\n", encoding="utf-8")
    redirected = tmp_path / "query_metrics_log.csv"

    monkeypatch.setattr(query_logging, "ENABLE_QUERY_LOG_CSV", True)
    monkeypatch.setattr(query_logging, "QUERY_LOG_CSV_PATH", str(protected))
    monkeypatch.setattr(query_logging, "DEFAULT_QUERY_LOG_CSV_PATH", redirected)
    monkeypatch.setattr(query_logging, "PROTECTED_QUERY_LOG_PATHS", {protected.resolve()})

    query_logging.log_query_row(
        input_query="test query",
        latency_ms=12.34,
        execution_path="deterministic",
        response_type="point",
        output="test answer",
    )

    assert protected.read_text(encoding="utf-8") == "keep as fixture\n"
    logged = redirected.read_text(encoding="utf-8")
    assert "input_query,latency_ms,path,type,output" in logged
    assert "test query,12.34,deterministic,point,test answer" in logged
