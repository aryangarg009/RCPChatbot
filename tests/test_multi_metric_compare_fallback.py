# tests/test_multi_metric_compare_fallback.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chat_service import process_question
from config import CSV_PATH
from date_io import load_data


def test_multi_metric_session_compare_is_consistent_with_or_without_context() -> None:
    df = load_data(CSV_PATH)
    query = "For patient 46 in game0, compare force and sparc between sessions 18 and sessions 21"

    out_no_context = process_question(query, df, context=None)
    assert out_no_context["type"] == "error"
    assert "Multiple metrics were requested in one session comparison" in out_no_context["answer"]

    primer = process_question(
        "how does sparc change from session 18 to session 21 for patient 46 in game0?",
        df,
        context=None,
    )
    ctx = primer.get("context")

    out_with_context = process_question(query, df, context=ctx)
    assert out_with_context["type"] == "error"
    assert "Multiple metrics were requested in one session comparison" in out_with_context["answer"]
