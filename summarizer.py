# summarizer.py
import math
import re
from typing import List, Dict, Any, Optional

from schema import QuerySpec

def _is_valid_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and math.isfinite(val)

def interpret_metric_change(metric_name: str, change: float) -> Optional[str]:
    """
    Return a short interpretation phrase (not a full sentence) for a given metric change.
    """
    if metric_name == "average_sparc":
        if change > 0:
            return "smoother, more continuous, better-coordinated movement"
        if change < 0:
            return "less smooth, jerkier, more interrupted movement"
        return "similar movement smoothness over this period"

    if metric_name == "avg_f_patient":
        if change > 0:
            return "increased strength output"
        if change < 0:
            return "reduced strength output"
        return "similar strength over this period"

    if metric_name == "avg_efficiency":
        if change > 0:
            return "improved hand-eye coordination accuracy"
        if change < 0:
            return "reduced hand-eye coordination accuracy"
        return "similar hand-eye coordination over this period"

    if metric_name == "area":
        if change > 0:
            return "increased range of motion"
        if change < 0:
            return "reduced range of motion"
        return "similar range of motion over this period"

    if metric_name == "timestampms":
        if change > 0:
            return "longer session duration"
        if change < 0:
            return "shorter session duration"
        return "similar session duration over this period"

    return None

def _metric_improvement_direction(metric_name: str) -> int:
    """
    Return +1 if higher is better, -1 if lower is better.
    Defaults to +1 for unknown metrics.
    """
    higher_better = {
        "average_sparc",  # closer to 0 (often higher) is better
        "avg_efficiency",
        "avg_f_patient",
        "area",
    }
    if metric_name in higher_better:
        return 1
    return 1

def classify_trend(per_date_summary: List[Dict[str, Any]], metric_name: str) -> Dict[str, Optional[str]]:
    """
    Classify overall trend as improving / worsening / variable / no clear trend.
    Uses step-to-step deltas with a small noise threshold.
    """
    if len(per_date_summary) < 2:
        return {"trend_label": None, "trend_reason": None}

    vals = [row["mean_metric_value"] for row in per_date_summary]
    if not vals:
        return {"trend_label": None, "trend_reason": None}

    baseline = vals[0]
    epsilon = max(1e-6, 0.01 * abs(baseline))
    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    direction = _metric_improvement_direction(metric_name)
    improving = 0
    worsening = 0
    for d in deltas:
        if d * direction > epsilon:
            improving += 1
        elif d * direction < -epsilon:
            worsening += 1

    total_nonflat = improving + worsening
    if total_nonflat == 0:
        return {
            "trend_label": "no clear trend",
            "trend_reason": "values stayed roughly stable between sessions",
        }

    improving_ratio = improving / total_nonflat
    worsening_ratio = worsening / total_nonflat

    if improving_ratio >= 0.8:
        return {
            "trend_label": "improving",
            "trend_reason": "values generally improved from session to session",
        }
    if worsening_ratio >= 0.8:
        return {
            "trend_label": "worsening",
            "trend_reason": "values generally worsened from session to session",
        }

    if improving > 0 and worsening > 0:
        return {
            "trend_label": "variable",
            "trend_reason": "values fluctuated with rises and drops between sessions",
        }

    # Fallback if only one direction but not strong enough for 80% rule
    label = "improving" if improving > 0 else "worsening"
    return {
        "trend_label": label,
        "trend_reason": (
            "values generally improved from session to session"
            if label == "improving"
            else "values generally worsened from session to session"
        ),
    }

def is_point_query(spec: QuerySpec, results: List[Dict[str, Any]]) -> bool:
    """
    True if:
    - explicit session, OR
    - exactly one unique date in results
    """
    if spec.session is not None:
        return True
    dates = {r.get("date") for r in results if "date" in r}
    return len(dates) == 1

def format_point_result(results: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    if len(results) == 1:
        r = results[0]
        if not _is_valid_number(r.get("metric_value")):
            return {"error": "No valid numeric values (missing/inf/invalid) found."}
        return {
            "type": "point",
            "metric": metric_name,
            "value": r["metric_value"],
            "date": r.get("date"),
            "session": r.get("session"),
            "game": r.get("game"),
        }
    
    games = {r.get("game") for r in results}
    if len(games) > 1:
        return {"error": f"Multiple games matched this session ({sorted(games)}). Please specify a game."}

    vals = [r["metric_value"] for r in results if _is_valid_number(r.get("metric_value"))]
    if not vals:
        return {"error": "No valid numeric values (missing/inf/invalid) found."}

    return {
        "type": "point",
        "metric": metric_name,
        "value": sum(vals) / len(vals),
        "n": len(vals),
        "date": results[0].get("date"),
        "session": results[0].get("session"),
        "game": results[0].get("game"),
    }

def summarize_timeseries(
    results: List[Dict[str, Any]],
    metric_name: str,
    requested_start: Optional[str] = None
) -> Dict[str, Any]:
    if len(results) == 1 and "error" in results[0]:
        return {"error": results[0]["error"]}

    by_date: Dict[str, List[float]] = {}
    for r in results:
        v = r.get("metric_value", None)
        if not _is_valid_number(v):
            continue
        by_date.setdefault(r["date"], []).append(float(v))

    if not by_date:
        return {"error": "No valid numeric values (missing/inf/invalid) found for this metric."}

    dates_sorted = sorted(by_date.keys())
    per_date = []
    for d in dates_sorted:
        vals = by_date[d]
        mean_v = sum(vals) / len(vals)
        per_date.append({
            "date": d,
            "n_sessions": len(vals),
            "mean_metric_value": mean_v,
            "min_metric_value": min(vals),
            "max_metric_value": max(vals),
        })

    first_date = per_date[0]["date"]
    last_date = per_date[-1]["date"]
    first_mean = per_date[0]["mean_metric_value"]
    last_mean = per_date[-1]["mean_metric_value"]
    change = last_mean - first_mean

    baseline_note = None
    if requested_start is not None and requested_start != first_date:
        baseline_note = (
            f"No data on requested start date {requested_start}; "
            f"using first available date {first_date} as baseline."
        )

    relative_change_pct = None
    if first_mean != 0:
        relative_change_pct = (change / abs(first_mean)) * 100

    interpretation = interpret_metric_change(metric_name, change)
    trend = classify_trend(per_date, metric_name)

    return {
        "per_date_summary": per_date,
        "change_first_to_last_mean": change,
        "relative_change_pct_vs_baseline": relative_change_pct,
        "first_date": first_date,
        "last_date": last_date,
        "baseline_note": baseline_note,
        "interpretation": interpretation,
        "trend_label": trend.get("trend_label"),
        "trend_reason": trend.get("trend_reason"),
    }

def summarize_session_range(
    results: List[Dict[str, Any]],
    metric_name: str,
    session_start: str,
    session_end: str
) -> Dict[str, Any]:
    if len(results) == 1 and "error" in results[0]:
        return {"error": results[0]["error"]}

    by_session: Dict[str, List[float]] = {}
    for r in results:
        v = r.get("metric_value", None)
        if not _is_valid_number(v):
            continue
        by_session.setdefault(r["session"], []).append(float(v))

    if not by_session:
        return {"error": "No valid numeric values (missing/inf/invalid) found for this metric."}

    def _session_num(s: Optional[str]) -> int:
        m = re.search(r"\bsession[_\s]*(\d+)\b", str(s).lower())
        return int(m.group(1)) if m else -1

    sessions_sorted = sorted(by_session.keys(), key=_session_num)
    per_session = []
    for s in sessions_sorted:
        vals = by_session[s]
        mean_v = sum(vals) / len(vals)
        per_session.append({
            "session": s,
            "n_rows": len(vals),
            "mean_metric_value": mean_v,
            "min_metric_value": min(vals),
            "max_metric_value": max(vals),
        })

    first_session = per_session[0]["session"]
    last_session = per_session[-1]["session"]
    first_mean = per_session[0]["mean_metric_value"]
    last_mean = per_session[-1]["mean_metric_value"]
    change = last_mean - first_mean

    baseline_note = None
    if session_start != first_session:
        baseline_note = (
            f"No data on requested start session {session_start}; "
            f"using first available session {first_session} as baseline."
        )

    relative_change_pct = None
    if first_mean != 0:
        relative_change_pct = (change / abs(first_mean)) * 100

    interpretation = interpret_metric_change(metric_name, change)
    trend = classify_trend(per_session, metric_name)

    return {
        "per_session_summary": per_session,
        "change_first_to_last_mean": change,
        "relative_change_pct_vs_baseline": relative_change_pct,
        "first_session": first_session,
        "last_session": last_session,
        "baseline_note": baseline_note,
        "interpretation": interpretation,
        "trend_label": trend.get("trend_label"),
        "trend_reason": trend.get("trend_reason"),
    }
