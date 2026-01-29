# query_engine.py
import math
import re
from typing import List, Dict, Any, Optional

import pandas as pd

from schema import QuerySpec

def extract_sessions_from_text(question: str) -> List[str]:
    """
    Returns a list like ["session_2", "session_10"] if user mentions them.
    Accepts 'session 2' or 'session_2'.
    """
    q = question.lower()
    matches = re.findall(r"\bsession[_\s]*(\d+)\b", q)
    sessions = []
    for m in matches:
        sessions.append(f"session_{int(m)}")
    return sessions

def detect_relative_session_cue(question: str) -> Optional[str]:
    """
    Detect relative session references like "first session", "previous session", etc.
    Returns one of: "first", "previous", "next", "latest", or None.
    """
    q = question.lower()
    if "first session" in q or "earliest session" in q:
        return "first"
    if "latest session" in q or "last session" in q or "most recent session" in q:
        return "latest"
    if "previous session" in q or "prior session" in q or "session before" in q:
        return "previous"
    if "next session" in q or "following session" in q or "session after" in q:
        return "next"
    return None

def resolve_relative_session(
    df: pd.DataFrame,
    base_spec: QuerySpec,
    cue: str
) -> Dict[str, Any]:
    """
    Resolve a relative session reference to an explicit session_id.
    Requires patient_id and game from base_spec. Uses session number ordering.
    """
    if base_spec.session is None:
        return {"error": "No base session in context to compare from."}
    if base_spec.patient_id == "__MISSING__":
        return {"error": "No patient in context to compare."}
    if base_spec.game is None:
        return {"error": "No game in context to compare."}

    def _session_num(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        m = re.search(r"\bsession[_\s]*(\d+)\b", str(s).lower())
        return int(m.group(1)) if m else None

    base_num = _session_num(base_spec.session)
    if base_num is None:
        return {"error": f"Could not parse session number from '{base_spec.session}'."}

    subset = df.copy()
    subset = subset[subset["patient_id"].astype(str).str.strip() == base_spec.patient_id]
    subset = subset[subset["game"].astype(str).str.strip() == base_spec.game]

    sessions = sorted({s for s in subset["session"].astype(str).str.strip() if s})
    numbered = []
    for s in sessions:
        n = _session_num(s)
        if n is not None:
            numbered.append((n, s))

    if not numbered:
        return {"error": "No comparable sessions found for that patient/game."}

    numbered.sort(key=lambda x: x[0])
    nums = [n for n, _ in numbered]
    sessions_by_num = {n: s for n, s in numbered}

    if cue == "first":
        return {"session": sessions_by_num[nums[0]]}
    if cue == "latest":
        return {"session": sessions_by_num[nums[-1]]}
    if cue == "previous":
        prev_nums = [n for n in nums if n < base_num]
        if not prev_nums:
            return {"error": "No previous session found before the current session."}
        return {"session": sessions_by_num[prev_nums[-1]]}
    if cue == "next":
        next_nums = [n for n in nums if n > base_num]
        if not next_nums:
            return {"error": "No next session found after the current session."}
        return {"session": sessions_by_num[next_nums[0]]}

    return {"error": "Unrecognized relative session cue."}

def normalize_session_string(s: str) -> Optional[str]:
    """
    Accepts 'session_2' or 'session 2' or 'Session 2' and returns 'session_2'.
    Returns None if it can't parse.
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    m = re.search(r"\bsession[_\s]*(\d+)\b", s)
    if not m:
        return None
    return f"session_{int(m.group(1))}"

def session_number(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    m = re.search(r"\bsession[_\s]*(\d+)\b", str(s).lower())
    return int(m.group(1)) if m else None

def _safe_metric_value(val: Any) -> Optional[float]:
    if pd.isna(val):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f

def run_query(df: pd.DataFrame, spec: QuerySpec) -> List[Dict[str, Any]]:
    missing = []
    if spec.patient_id == "__MISSING__":
        missing.append("patient_id")
    if spec.metric == "__MISSING__":
        missing.append("metric")

    needs_date_range = (spec.session is None)

    # If query is session-based (no dates), require BOTH session and game
    if spec.session is not None:
        if spec.game is None:
            return [{"error": "For session-based queries, please specify a game (e.g., 'game0') because the same session can exist in multiple games."}]
            # Also ensure dates are not provided (optional strictness)
            # if spec.date_start != "__MISSING__" or spec.date_end != "__MISSING__":
            # return [{"error": "For session queries, don't provide dates (or use a date-range query instead)."}]

    # Require game for date-range queries (session is None)
    if spec.session is None:
        if spec.game is None:
            return [{"error": "Please specify a game (e.g., 'game0') for date-range queries to avoid mixing data across games."}]
    
    if needs_date_range and spec.date_start == "__MISSING__":
        missing.append("date_start")

    if missing:
        return [{"error": f"Missing required info: {', '.join(missing)}"}]

    if spec.metric not in df.columns:
        return [{"error": f"Metric column '{spec.metric}' not found in CSV."}]

    out = df.copy()
    out = out[out["patient_id"].astype(str).str.strip() == spec.patient_id]

    if spec.game is not None:
        out = out[out["game"].astype(str).str.strip() == spec.game]
    if spec.session is not None:
        out = out[out["session"].astype(str).str.strip() == spec.session]

    if spec.date_start != "__MISSING__" and spec.date_end != "__MISSING__":
        start_dt = pd.to_datetime(spec.date_start, errors="raise")
        end_dt = pd.to_datetime(spec.date_end, errors="raise")
        out = out[(out["date"] >= start_dt) & (out["date"] <= end_dt)]

    out = out.sort_values(["date", "game", "session"])

    rows = []
    for _, row in out.iterrows():
        rows.append({
            "date": row["date"].date().isoformat() if pd.notna(row["date"]) else None,
            "patient_id": row["patient_id"],
            "metric_value": _safe_metric_value(row[spec.metric]),
            "game": row.get("game", None),
            "session": row.get("session", None),
        })

    if not rows:
        return [{"error": "No matching rows found in uploaded CSV for that query."}]

    return rows

def run_session_range(
    df: pd.DataFrame,
    spec: QuerySpec,
    session_start: str,
    session_end: str
) -> List[Dict[str, Any]]:
    """
    Query rows for a patient/game/metric across a continuous session range.
    """
    if spec.patient_id == "__MISSING__":
        return [{"error": "Missing required info: patient_id"}]
    if spec.metric == "__MISSING__":
        return [{"error": "Missing required info: metric"}]
    if spec.game is None:
        return [{"error": "Please specify a game (e.g., 'game0') for session-range queries."}]

    start_num = session_number(session_start)
    end_num = session_number(session_end)
    if start_num is None or end_num is None:
        return [{"error": "Could not parse session range. Use 'session 1 to session 7'."}]

    lo, hi = (start_num, end_num) if start_num <= end_num else (end_num, start_num)

    out = df.copy()
    out = out[out["patient_id"].astype(str).str.strip() == spec.patient_id]
    out = out[out["game"].astype(str).str.strip() == spec.game]

    def in_range(s: str) -> bool:
        n = session_number(s)
        return n is not None and lo <= n <= hi

    out = out[out["session"].astype(str).str.strip().apply(in_range)]
    out = out.sort_values(["session", "date"])

    rows = []
    for _, row in out.iterrows():
        rows.append({
            "date": row["date"].date().isoformat() if pd.notna(row["date"]) else None,
            "patient_id": row["patient_id"],
            "metric_value": _safe_metric_value(row[spec.metric]),
            "game": row.get("game", None),
            "session": row.get("session", None),
        })

    if not rows:
        return [{"error": "No matching rows found in uploaded CSV for that query."}]

    return rows

def mean_metric_value(rows: List[Dict[str, Any]]) -> Optional[float]:
    vals = [
        r["metric_value"]
        for r in rows
        if isinstance(r.get("metric_value"), (int, float)) and math.isfinite(r["metric_value"])
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)

def compare_two_sessions(
    df: pd.DataFrame,
    base_spec: QuerySpec,
    other_session: str
) -> Dict[str, Any]:
    """
    Compare base_spec.session vs other_session for the SAME patient/game/metric.
    Dates are ignored (session-specific queries).
    """
    if base_spec.session is None:
        return {"error": "No base session in context to compare from."}
    if base_spec.metric == "__MISSING__":
        return {"error": "No metric in context to compare."}
    if base_spec.patient_id == "__MISSING__":
        return {"error": "No patient in context to compare."}

    spec_a = base_spec.model_copy(deep=True)
    spec_b = base_spec.model_copy(deep=True)

    spec_a.date_start = "__MISSING__"
    spec_a.date_end = "__MISSING__"
    spec_b.date_start = "__MISSING__"
    spec_b.date_end = "__MISSING__"

    spec_a.session = base_spec.session
    spec_b.session = other_session

    rows_a = run_query(df, spec_a)
    if len(rows_a) == 1 and "error" in rows_a[0]:
        return {"error": f"Could not fetch {spec_a.session}: {rows_a[0]['error']}"}

    rows_b = run_query(df, spec_b)
    if len(rows_b) == 1 and "error" in rows_b[0]:
        return {"error": f"Could not fetch {spec_b.session}: {rows_b[0]['error']}"}

    mean_a = mean_metric_value(rows_a)
    mean_b = mean_metric_value(rows_b)
    if mean_a is None or mean_b is None:
        return {"error": "One of the sessions has no numeric metric values."}

    def _min_date(rows: List[Dict[str, Any]]) -> Optional[pd.Timestamp]:
        dates = [pd.to_datetime(r.get("date"), errors="coerce") for r in rows]
        dates = [d for d in dates if pd.notna(d)]
        return min(dates) if dates else None

    num_a = session_number(spec_a.session)
    num_b = session_number(spec_b.session)
    date_a = _min_date(rows_a)
    date_b = _min_date(rows_b)

    # Order by session number when possible; otherwise by date; else keep base -> other
    earlier_session = spec_a.session
    later_session = spec_b.session
    earlier_value = mean_a
    later_value = mean_b
    rows_earlier = rows_a
    rows_later = rows_b

    if num_a is not None and num_b is not None and num_a != num_b:
        if num_a > num_b:
            earlier_session, later_session = later_session, earlier_session
            earlier_value, later_value = later_value, earlier_value
            rows_earlier, rows_later = rows_later, rows_earlier
    elif date_a is not None and date_b is not None and date_a != date_b:
        if date_a > date_b:
            earlier_session, later_session = later_session, earlier_session
            earlier_value, later_value = later_value, earlier_value
            rows_earlier, rows_later = rows_later, rows_earlier

    change_later_minus_earlier = later_value - earlier_value
    diff_earlier_minus_later = earlier_value - later_value
    relative_change_pct = None
    if earlier_value != 0:
        relative_change_pct = (change_later_minus_earlier / abs(earlier_value)) * 100

    return {
        "patient_id": base_spec.patient_id,
        "game": base_spec.game,
        "metric": base_spec.metric,
        "session_a": spec_a.session,
        "session_b": spec_b.session,
        "value_a": mean_a,
        "value_b": mean_b,
        "session_earlier": earlier_session,
        "session_later": later_session,
        "value_earlier": earlier_value,
        "value_later": later_value,
        "change_later_minus_earlier": change_later_minus_earlier,
        "diff_earlier_minus_later": diff_earlier_minus_later,
        "relative_change_pct_vs_earlier": relative_change_pct,
        "rows_a": rows_a,
        "rows_b": rows_b,
        "rows_earlier": rows_earlier,
        "rows_later": rows_later,
    }
