# narration.py
from typing import Dict, Any, Optional

from summarizer import interpret_metric_change

def _choose_hedge_phrase(relative_change_pct: Optional[float]) -> str:
    """
    Pick a conservative hedging phrase based on effect size.
    Falls back to a neutral phrase if percent change is unavailable.
    """
    if relative_change_pct is None:
        return "is consistent with"
    magnitude = abs(relative_change_pct)
    if magnitude == 0:
        return "is consistent with"
    if magnitude < 5:
        return "may indicate"
    if magnitude < 15:
        return "is consistent with"
    return "suggests"

def metric_display_name(metric: str) -> str:
    names = {
        "average_sparc": "smoothness (SPARC)",
        "avg_efficiency": "efficiency",
        "avg_f_patient": "total force",
        "area": "range of motion (area)",
    }
    return names.get(metric, metric)

def _metric_label(metric: str) -> str:
    labels = {
        "average_sparc": "movement smoothness",
        "area": "range of motion",
        "avg_efficiency": "movement efficiency",
        "avg_f_patient": "applied force",
    }
    return labels.get(metric, metric)

def narrate_timeseries(summary: dict, spec) -> str:
    """
    Convert a summarize_timeseries() output into a natural-language explanation.
    Deterministic, no LLM.
    """

    if "error" in summary:
        return summary["error"]

    metric = spec.metric
    start = summary["first_date"]
    end = summary["last_date"]
    change = summary["change_first_to_last_mean"]
    pct = summary.get("relative_change_pct_vs_baseline")
    interpretation = summary.get("interpretation")
    trend_label = summary.get("trend_label")
    trend_reason = summary.get("trend_reason")

    # Friendly metric names
    metric_label = _metric_label(metric)

    # Direction language
    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "remained stable"

    # Opening sentence
    text = (
        f"I found patient records from {start} to {end}. "
    )

    # Change sentence
    text += (
        f"Over this period, the average {metric_label} "
        f"{direction} by {abs(change):.4f}"
    )

    if pct is not None:
        text += f", which corresponds to a {abs(pct):.2f}% change from the baseline."

    else:
        text += "."

    # Baseline note (if present)
    if summary.get("baseline_note"):
        text += " " + summary["baseline_note"]

    # Interpretation sentence (hedged)
    if interpretation:
        hedge = _choose_hedge_phrase(pct)
        interp = interpretation.strip().rstrip(".")
        text += f" This change {hedge} {interp}."

    if trend_label:
        if trend_label == "variable":
            trend_text = "Overall, the trend shows fluctuations with rises and drops between sessions"
            if change > 0:
                trend_text += ", but it increased overall from the first to the last date"
            elif change < 0:
                trend_text += ", but it decreased overall from the first to the last date"
            else:
                trend_text += ", ending near the starting level"
        elif trend_label == "no clear trend":
            trend_text = "Overall, values stayed roughly stable between sessions"
        else:
            trend_text = "Overall, values generally improved from session to session" if trend_label == "improving" else "Overall, values generally worsened from session to session"
        if trend_reason and trend_label in {"improving", "worsening", "no clear trend"}:
            trend_text = trend_reason.capitalize() if trend_reason else trend_text
        text += f" {trend_text}."

    return text

def narrate_point(point: Dict[str, Any], metric_name: str, patient_id: str) -> str:
    """
    Deterministic natural-language narration for a single-point result.
    """
    if "error" in point:
        return f"I couldn’t compute that result: {point['error']}"

    metric_label = metric_display_name(metric_name)
    value = point.get("value")
    date = point.get("date")
    session = point.get("session")
    game = point.get("game")

    parts = [f"For patient {patient_id}, the {metric_label} value is {value:.6f}"]

    if session and game and date:
        parts.append(f"in {game}, {session} on {date}.")
    elif session and game:
        parts.append(f"in {game}, {session}.")
    elif game and date:
        parts.append(f"in {game} on {date}.")
    elif date:
        parts.append(f"on {date}.")
    else:
        parts.append("for the record I found.")

    return " ".join(parts)

def narrate_session_comparison(cmp_out: Dict[str, Any]) -> str:
    """
    Deterministic narration for comparing two sessions (earlier -> later).
    """
    if "error" in cmp_out:
        return cmp_out["error"]

    patient_id = cmp_out["patient_id"]
    game = cmp_out["game"]
    metric = cmp_out["metric"]
    session_earlier = cmp_out["session_earlier"]
    session_later = cmp_out["session_later"]
    value_earlier = cmp_out["value_earlier"]
    value_later = cmp_out["value_later"]
    change = cmp_out["change_later_minus_earlier"]
    diff_earlier_minus_later = cmp_out["diff_earlier_minus_later"]
    pct = cmp_out.get("relative_change_pct_vs_earlier")

    metric_label = _metric_label(metric)

    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "remained stable"

    text = (
        f"For patient {patient_id} in {game}, comparing {session_earlier} to {session_later}, "
        f"the average {metric_label} {direction} by {abs(change):.4f} "
        f"(from {value_earlier:.4f} to {value_later:.4f})."
    )

    text += f" The difference (earlier - later) is {diff_earlier_minus_later:.4f}."

    if pct is not None:
        text += f" This corresponds to a {abs(pct):.2f}% change relative to the earlier session."

    interpretation = interpret_metric_change(metric, change)
    if interpretation:
        hedge = _choose_hedge_phrase(pct)
        text += f" This change {hedge} {interpretation}."

    return text

def narrate_session_range(summary: dict, spec, session_start: str, session_end: str) -> str:
    """
    Narration for a continuous session range (session start -> session end).
    """
    if "error" in summary:
        return summary["error"]

    metric_label = _metric_label(spec.metric)
    start = summary["first_session"]
    end = summary["last_session"]
    change = summary["change_first_to_last_mean"]
    pct = summary.get("relative_change_pct_vs_baseline")
    interpretation = summary.get("interpretation")

    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "remained stable"

    text = (
        f"I found patient records from {start} to {end}. "
        f"Over this session range, the average {metric_label} {direction} by {abs(change):.4f}"
    )

    if pct is not None:
        text += f", which corresponds to a {abs(pct):.2f}% change from the baseline."
    else:
        text += "."

    if summary.get("baseline_note"):
        text += " " + summary["baseline_note"]

    if interpretation:
        hedge = _choose_hedge_phrase(pct)
        interp = interpretation.strip().rstrip(".")
        text += f" This change {hedge} {interp}."

    trend_label = summary.get("trend_label")
    trend_reason = summary.get("trend_reason")
    if trend_label:
        if trend_label == "variable":
            trend_text = "Overall, the trend shows fluctuations with rises and drops between sessions"
            if change > 0:
                trend_text += ", but it increased overall from the first to the last session"
            elif change < 0:
                trend_text += ", but it decreased overall from the first to the last session"
            else:
                trend_text += ", ending near the starting level"
        elif trend_label == "no clear trend":
            trend_text = "Values stayed roughly stable between sessions"
        else:
            trend_text = "Values generally improved from session to session" if trend_label == "improving" else "Values generally worsened from session to session"
        if trend_reason and trend_label in {"improving", "worsening", "no clear trend"}:
            trend_text = trend_reason.capitalize()
        text += f" {trend_text}."

    return text
