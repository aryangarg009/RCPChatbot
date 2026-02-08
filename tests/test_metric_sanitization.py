# tests/test_metric_sanitization.py
import pandas as pd

from schema import QuerySpec
from query_engine import run_query
from summarizer import summarize_timeseries


def test_invalid_metric_values_are_ignored() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]
            ),
            "patient": ["P1", "P1", "P1", "P1"],
            "game": ["game0"] * 4,
            "session": ["session_1", "session_2", "session_3", "session_4"],
            "area": [1.0, "inf", "#NAME", None],
        }
    )

    spec = QuerySpec(
        action="get_metric_timeseries",
        patient="P1",
        metric="area",
        date_start="2022-01-01",
        date_end="2022-01-04",
        game="game0",
        session=None,
        return_columns=["date", "patient", "metric_value"],
    )

    results = run_query(df, spec)
    assert results[0]["metric_value"] == 1.0
    assert results[1]["metric_value"] is None
    assert results[2]["metric_value"] is None
    assert results[3]["metric_value"] is None


def test_all_invalid_values_returns_clear_error() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-01-02"]),
            "patient": ["P1", "P1"],
            "game": ["game0", "game0"],
            "session": ["session_1", "session_2"],
            "area": ["inf", "#NAME"],
        }
    )

    spec = QuerySpec(
        action="get_metric_timeseries",
        patient="P1",
        metric="area",
        date_start="2022-01-01",
        date_end="2022-01-02",
        game="game0",
        session=None,
        return_columns=["date", "patient", "metric_value"],
    )

    results = run_query(df, spec)
    summary = summarize_timeseries(results, metric_name="area", requested_start=spec.date_start)
    assert summary["error"] == "No valid numeric values (missing/inf/invalid) found for this metric."
