# narration.py
import math
from typing import Any, Dict, List, Optional

from summarizer import interpret_metric_change

CLINICAL_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "average_sparc": {
        "descending": True,
        "interval": 0.5,
        "bands": [
            {
                "name": "Optimal",
                "start": -1.6,
                "adl": "Neat handwriting with smooth, controlled pen movement.",
            },
            {
                "name": "Near Optimal",
                "start": -2.1,
                "adl": "Legible writing with slight unevenness; minor wobble when drawing lines.",
            },
            {
                "name": "Mild",
                "start": -2.6,
                "adl": "Mostly legible writing with occasional shakiness and small control breaks.",
            },
            {
                "name": "Moderate",
                "start": -3.1,
                "adl": "Noticeably uneven writing; letters become inconsistent and fine control drops.",
            },
            {
                "name": "Significant",
                "start": -3.6,
                "adl": "Messier writing that is harder to read and requires more effort.",
            },
            {
                "name": "Severe",
                "start": -4.1,
                "adl": "Very shaky hand output with frequent distortions during precision tasks.",
            },
            {
                "name": "Very Severe",
                "start": -4.6,
                "adl": "Barely legible writing; words can break apart during continuous motion.",
            },
            {
                "name": "Extreme",
                "start": -5.1,
                "adl": "Illegible handwriting with major difficulty in controlled hand movement.",
            },
        ],
        "games": [("Explore the World", 0), ("Restaurant", 5)],
        "low_actions": [
            "prioritize H-Man games Explore the World (game 0) and Restaurant (game 5) with slower, accuracy-first rounds.",
            "In therapy, focus on line tracing, small circles, and short handwriting blocks before increasing speed.",
        ],
        "maintain_action": (
            "maintain current fine-motor progression with Explore the World (game 0) and "
            "Restaurant (game 5), tightening precision goals gradually."
        ),
    },
    "avg_f_patient": {
        "descending": False,
        "interval": 8.0,
        "bands": [
            {
                "name": "Optimal",
                "start": -30.0,
                "adl": "About 4 iPad-equivalent force; daily tasks like lifting a full water bottle or heavier grocery loads are typically manageable.",
            },
            {
                "name": "Near Optimal",
                "start": -22.0,
                "adl": "About 3 iPad-equivalent force; moderate household loads are usually manageable.",
            },
            {
                "name": "Mild",
                "start": -14.0,
                "adl": "About 2 iPad-equivalent force; light-to-moderate objects are manageable with effort.",
            },
            {
                "name": "Moderate",
                "start": -6.0,
                "adl": "About 1 iPad-equivalent force; mainly lighter objects are practical in daily tasks.",
            },
            {
                "name": "Significant",
                "start": 2.0,
                "adl": "Around 1 iPad-equivalent force with limited reserve; heavier carrying tasks may need support.",
            },
            {
                "name": "Severe",
                "start": 10.0,
                "adl": "Around 2 iPad-equivalent force but with poor consistency; lifting and carrying can still be effortful.",
            },
            {
                "name": "Very Severe",
                "start": 18.0,
                "adl": "Around 3 iPad-equivalent force with marked control issues during household handling.",
            },
            {
                "name": "Extreme",
                "start": 26.0,
                "adl": "Around 4 iPad-equivalent force with major control variability during daily lifting.",
            },
        ],
        "games": [("Drone", 2), ("Matching Pairs", 8)],
        "low_actions": [
            "prioritize H-Man games Drone (game 2) and Matching Pairs (game 8) with graded resistance targets.",
            "In therapy, use progressive grip and lift practice (water bottle to grocery bag loads) with form cues and rest breaks.",
        ],
        "maintain_action": (
            "maintain graded strengthening in Drone (game 2) and Matching Pairs (game 8), "
            "progressing load and endurance in small steps."
        ),
    },
    "avg_efficiency": {
        "descending": True,
        "interval": 0.0625,
        "bands": [
            {
                "name": "Optimal",
                "start": 0.95,
                "adl": "Straight movement path, like drawing a clean line between targets.",
            },
            {
                "name": "Near Optimal",
                "start": 0.8875,
                "adl": "Almost straight movement path with only minor detours.",
            },
            {
                "name": "Mild",
                "start": 0.825,
                "adl": "Noticeable detours; hand path is less direct during reach tasks.",
            },
            {
                "name": "Moderate",
                "start": 0.7625,
                "adl": "Curved hand path with extra steps before reaching the target.",
            },
            {
                "name": "Significant",
                "start": 0.7,
                "adl": "Roundabout pathing that can slow self-care movements.",
            },
            {
                "name": "Severe",
                "start": 0.6375,
                "adl": "Frequent wandering from the intended path during controlled movement.",
            },
            {
                "name": "Very Severe",
                "start": 0.575,
                "adl": "Looping movement patterns that reduce precision and efficiency.",
            },
            {
                "name": "Extreme",
                "start": 0.4,
                "adl": "Path appears lost/disorganized, affecting directed hand function.",
            },
        ],
        "games": [("Race Car", 6), ("Flower Shop", 7)],
        "low_actions": [
            "prioritize H-Man games Race Car (game 6) and Flower Shop (game 7) with slower path-tracking targets.",
            "In therapy, focus on reach-to-target drills with visual waypoints and controlled stop-start transitions.",
        ],
        "maintain_action": (
            "maintain coordination progression in Race Car (game 6) and Flower Shop (game 7), "
            "increasing path complexity while preserving accuracy."
        ),
    },
}


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


def _format_duration_ms(value: Optional[float]) -> str:
    if value is None:
        return "unknown duration"
    try:
        total_seconds = int(round(abs(float(value)) / 1000.0))
    except (TypeError, ValueError):
        return "unknown duration"
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sign = "-" if value < 0 else ""
    if hours > 0:
        return f"{sign}{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{sign}{minutes}m {seconds}s"
    return f"{sign}{seconds}s"


def _format_metric_value(metric: str, value: Optional[float]) -> str:
    if metric == "timestampms":
        return _format_duration_ms(value)
    if value is None:
        return "nan"
    return f"{value:.6f}"


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _change_text(metric: str, change: float) -> str:
    if metric == "timestampms":
        return _format_duration_ms(abs(change))
    return f"{abs(change):.4f}"


def _direction_word(change: float) -> str:
    if change > 0:
        return "increased"
    if change < 0:
        return "decreased"
    return "remained stable"


def metric_display_name(metric: str) -> str:
    names = {
        "average_sparc": "smoothness (SPARC)",
        "avg_efficiency": "efficiency",
        "avg_f_patient": "total force",
        "area": "range of motion (area)",
        "timestampms": "session duration",
    }
    return names.get(metric, metric)


def _metric_label(metric: str) -> str:
    labels = {
        "average_sparc": "movement smoothness",
        "area": "range of motion",
        "avg_efficiency": "movement efficiency",
        "avg_f_patient": "applied force",
        "timestampms": "session duration",
    }
    return labels.get(metric, metric)


def _classify_clinical_band(metric: str, value: Optional[float]) -> Optional[Dict[str, Any]]:
    cfg = CLINICAL_GUIDANCE.get(metric)
    val = _to_finite_float(value)
    if cfg is None or val is None:
        return None

    bands = cfg["bands"]
    thresholds = [float(b["start"]) for b in bands]
    descending = bool(cfg["descending"])
    idx = len(bands) - 1

    if descending:
        if val > thresholds[0]:
            idx = 0
        elif val <= thresholds[-1]:
            idx = len(bands) - 1
        else:
            for i in range(len(thresholds) - 1):
                upper = thresholds[i]
                lower_next = thresholds[i + 1]
                # Interval rule from provided sheet: current threshold to (not including) next threshold.
                if val <= upper and val > lower_next:
                    idx = i
                    break
    else:
        if val < thresholds[0]:
            idx = 0
        elif val >= thresholds[-1]:
            idx = len(bands) - 1
        else:
            for i in range(len(thresholds) - 1):
                lower = thresholds[i]
                upper_next = thresholds[i + 1]
                # Interval rule from provided sheet: current threshold to (not including) next threshold.
                if val >= lower and val < upper_next:
                    idx = i
                    break

    band = dict(bands[idx])
    band["index"] = idx
    return band


def _is_stagnating(metric: str, trend_label: Optional[str], change: Optional[float]) -> bool:
    if trend_label in {"no clear trend", "variable"}:
        return True
    if change is None:
        return False
    cfg = CLINICAL_GUIDANCE.get(metric)
    if cfg is None:
        return False
    epsilon = float(cfg.get("interval", 0.0)) * 0.20
    if epsilon <= 0:
        return False
    return abs(change) < epsilon


def _daily_life_impact(metric: str, current_value: Optional[float], band: Optional[Dict[str, Any]]) -> str:
    if band is not None:
        return str(band["adl"]).strip().rstrip(".")
    if metric == "area":
        return (
            "ROM-specific daily-life mapping is not configured yet; "
            "ROM analogies will be added once bounds are provided."
        )
    if metric == "timestampms":
        return (
            "Session duration is a workload/endurance context metric, "
            "not a direct movement-quality daily-life scale."
        )
    return "No daily-life mapping is configured yet for this metric."


def _action_suggestions(
    metric: str,
    band: Optional[Dict[str, Any]],
    trend_label: Optional[str],
    change: Optional[float],
) -> str:
    cfg = CLINICAL_GUIDANCE.get(metric)
    if cfg is None:
        if metric == "area":
            return (
                "ROM action mapping is pending. Once ROM bounds are added, "
                "this section will include targeted H-Man games and therapy priorities."
            )
        if metric == "timestampms":
            return (
                "Use duration with SPARC, force, and path efficiency results "
                "to calibrate workload; no direct low/stagnant action rule is set for duration alone."
            )
        return "No metric-specific action map is configured yet."

    low_or_concerning = band is not None and int(band["index"]) >= 2
    stagnating = _is_stagnating(metric, trend_label, change)
    if low_or_concerning or stagnating:
        low_actions = [str(x).strip() for x in cfg.get("low_actions", []) if str(x).strip()]
        if low_actions:
            if len(low_actions) == 1:
                return low_actions[0].rstrip(".")
            return f"{low_actions[0].rstrip('.')}. {low_actions[1].rstrip('.')}"
        return "prioritize targeted motor retraining"

    maintain = str(cfg.get("maintain_action", "")).strip()
    return maintain.rstrip(".") or "continue the current program and reassess at the next session"


def _clinical_interpretation_paragraph(
    metric: str,
    current_value: Optional[float],
    trend_label: Optional[str],
    change: Optional[float],
) -> str:
    band = _classify_clinical_band(metric, current_value)
    daily_life = _daily_life_impact(metric, current_value, band)
    action = _action_suggestions(metric, band, trend_label, change)
    if band is not None:
        band_sentence = f" This is in the {band['name']} band."
    else:
        band_sentence = ""
    return (
        f"{band_sentence} In daily life, this equates to: {daily_life.rstrip('.')}. "
        f"It is suggested to {action.rstrip('.')}."
    )


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

    metric_label = _metric_label(metric)
    direction = _direction_word(change)

    text = f"I found patient records from {start} to {end}. "
    text += (
        f"Over this period, the average {metric_label} "
        f"{direction} by {_change_text(metric, change)}"
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
            trend_text = (
                "Overall, values generally improved from session to session"
                if trend_label == "improving"
                else "Overall, values generally worsened from session to session"
            )
        if trend_reason and trend_label in {"improving", "worsening", "no clear trend"}:
            trend_text = trend_reason.capitalize() if trend_reason else trend_text
        text += f" {trend_text}."

    latest_value = None
    per_date_summary = summary.get("per_date_summary") or []
    if per_date_summary:
        latest_value = per_date_summary[-1].get("mean_metric_value")

    text += _clinical_interpretation_paragraph(
        metric=metric,
        current_value=latest_value,
        trend_label=trend_label,
        change=change,
    )
    return text


def narrate_point(point: Dict[str, Any], metric_name: str, patient: str) -> str:
    """
    Deterministic natural-language narration for a single-point result.
    """
    if "error" in point:
        return f"I couldn't compute that result: {point['error']}"

    metric_label = metric_display_name(metric_name)
    value = point.get("value")
    date = point.get("date")
    session = point.get("session")
    game = point.get("game")

    if metric_name == "timestampms":
        value_text = _format_metric_value(metric_name, value)
        parts = [f"For patient {patient}, the {metric_label} is {value_text}"]
    else:
        parts = [f"For patient {patient}, the {metric_label} value is {_format_metric_value(metric_name, value)}"]

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

    text = " ".join(parts)
    text += _clinical_interpretation_paragraph(
        metric=metric_name,
        current_value=_to_finite_float(value),
        trend_label=None,
        change=None,
    )
    return text


def narrate_session_comparison(cmp_out: Dict[str, Any]) -> str:
    """
    Deterministic narration for comparing two sessions (earlier -> later).
    """
    if "error" in cmp_out:
        return cmp_out["error"]

    patient = cmp_out["patient"]
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
    direction = _direction_word(change)

    if metric == "timestampms":
        earlier_text = _format_metric_value(metric, value_earlier)
        later_text = _format_metric_value(metric, value_later)
        text = (
            f"For patient {patient} in {game}, comparing {session_earlier} to {session_later}, "
            f"the average {metric_label} {direction} by {_change_text(metric, change)} "
            f"(from {earlier_text} to {later_text})."
        )
    else:
        text = (
            f"For patient {patient} in {game}, comparing {session_earlier} to {session_later}, "
            f"the average {metric_label} {direction} by {abs(change):.4f} "
            f"(from {value_earlier:.4f} to {value_later:.4f})."
        )

    if metric == "timestampms":
        diff_text = _format_duration_ms(diff_earlier_minus_later)
        text += f" The difference (earlier - later) is {diff_text}."
    else:
        text += f" The difference (earlier - later) is {diff_earlier_minus_later:.4f}."

    if pct is not None:
        text += f" This corresponds to a {abs(pct):.2f}% change relative to the earlier session."

    interpretation = interpret_metric_change(metric, change)
    if interpretation:
        hedge = _choose_hedge_phrase(pct)
        text += f" This change {hedge} {interpretation}."

    text += _clinical_interpretation_paragraph(
        metric=metric,
        current_value=_to_finite_float(value_later),
        trend_label=None,
        change=change,
    )
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

    direction = _direction_word(change)

    text = (
        f"I found patient records from {start} to {end}. "
        f"Over this session range, the average {metric_label} {direction} by {_change_text(spec.metric, change)}"
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
            trend_text = (
                "Values generally improved from session to session"
                if trend_label == "improving"
                else "Values generally worsened from session to session"
            )
        if trend_reason and trend_label in {"improving", "worsening", "no clear trend"}:
            trend_text = trend_reason.capitalize()
        text += f" {trend_text}."

    latest_value = None
    per_session_summary = summary.get("per_session_summary") or []
    if per_session_summary:
        latest_value = per_session_summary[-1].get("mean_metric_value")

    text += _clinical_interpretation_paragraph(
        metric=spec.metric,
        current_value=latest_value,
        trend_label=trend_label,
        change=change,
    )
    return text
