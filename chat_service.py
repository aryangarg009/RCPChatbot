# chat_service.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

from config import (
    ALLOWED_GAMES,
    ALLOWED_METRICS,
    ALLOWED_SESSIONS,
    CSV_PATH,
    ENABLE_CODE_FALLBACK,
    RESET_COMMANDS,
)
from context import (
    apply_followup_context,
    extract_patient_from_text,
    extract_metric_from_text,
    extract_metric_or_alias_from_definition_question,
    is_metric_definition_question,
    is_gender_question,
    question_mentions_dates,
    question_mentions_game,
    question_mentions_patient,
    question_mentions_session,
)
from llm_client import llm_question_to_query, deterministic_question_to_query
from narration import (
    narrate_point,
    narrate_session_comparison,
    narrate_session_range,
    narrate_timeseries,
)
from metrics import METRIC_EXPLANATIONS
from query_engine import (
    compare_two_sessions,
    detect_relative_session_cue,
    extract_sessions_from_text,
    resolve_relative_session,
    run_query,
    run_session_range,
)
from schema import QuerySpec
from summarizer import (
    format_point_result,
    is_point_query,
    summarize_session_range,
    summarize_timeseries,
)
from openai_fallback import OpenAIFallbackError, run_code_fallback


def _is_session_range_question(text: str) -> bool:
    t = text.lower()
    return ("from session" in t and "to session" in t) or (
        "between session" in t and "and session" in t
    )


def _context_from_state(
    last_spec: Optional[QuerySpec],
    last_session_range: Optional[Tuple[str, str]],
) -> Dict[str, Any]:
    return {
        "last_spec": last_spec.model_dump() if last_spec is not None else None,
        "last_session_range": list(last_session_range) if last_session_range else None,
    }


def _state_from_context(context: Optional[Dict[str, Any]]) -> Tuple[Optional[QuerySpec], Optional[Tuple[str, str]]]:
    if not context:
        return None, None
    last_spec = None
    last_session_range = None
    raw_spec = context.get("last_spec")
    if raw_spec:
        try:
            last_spec = QuerySpec(**raw_spec)
        except ValidationError:
            last_spec = None
    raw_range = context.get("last_session_range")
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        last_session_range = (str(raw_range[0]), str(raw_range[1]))
    return last_spec, last_session_range


def process_question(question: str, df, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last_spec, last_session_range = _state_from_context(context)
    ql = question.strip().lower()

    if ql in RESET_COMMANDS:
        return {
            "type": "reset",
            "answer": "Context cleared. Ask a new question with patient/metric/date.",
            "data": None,
            "context": _context_from_state(None, None),
        }

    if is_gender_question(question):
        patient = extract_patient_from_text(question)
        if patient is None and last_spec is not None:
            patient = last_spec.patient
        if patient is None or patient == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a patient to look up their gender.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if "gender" not in df.columns:
            return {
                "type": "error",
                "answer": "Gender column not found in the CSV.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        subset = df[df["patient"].astype(str).str.strip() == str(patient).strip()]
        if subset.empty:
            return {
                "type": "error",
                "answer": "No matching rows found for that patient.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        genders = (
            subset["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"m": "male", "f": "female"})
        )
        genders = {g for g in genders if g not in {"", "nan", "none"}}
        if not genders:
            return {
                "type": "error",
                "answer": "No gender data found for that patient.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if len(genders) > 1:
            return {
                "type": "error",
                "answer": f"Conflicting gender values found for patient {patient}.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        gender = sorted(genders)[0]
        return {
            "type": "gender",
            "answer": f"Patient {patient} is {gender}.",
            "data": {"patient": patient, "gender": gender},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- METRIC DEFINITION MODE ----
    # Only treat as a definition question if the user isn't asking about
    # a specific patient/game/session/date.
    if (
        is_metric_definition_question(question)
        and not question_mentions_patient(question)
        and not question_mentions_game(question)
        and not question_mentions_session(question)
        and not question_mentions_dates(question)
    ):
        metric = extract_metric_or_alias_from_definition_question(question)
        if metric is None:
            return {
                "type": "error",
                "answer": "I’m not sure which metric you mean. Try: 'what is sparc?' or 'what does efficiency mean?'",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

        explanation = METRIC_EXPLANATIONS.get(metric)
        if explanation is None:
            return {
                "type": "error",
                "answer": f"I don’t have an explanation written yet for '{metric}', but I can add one.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

        return {
            "type": "definition",
            "answer": explanation,
            "data": {"metric": metric},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- SESSION COMPARISON MODE (follow-up) ----
    if ("differ" in ql or "difference" in ql or "compare" in ql) and last_spec is not None:
        # If user explicitly mentions patient/metric/game, treat as standalone compare
        if (
            question_mentions_patient(question)
            or question_mentions_game(question)
            or extract_metric_from_text(question) is not None
        ):
            pass
        else:
            sessions = extract_sessions_from_text(question)
            if not sessions:
                cue = detect_relative_session_cue(question)
                if cue is None:
                    return {
                        "type": "error",
                        "answer": "Please mention the session number to compare (e.g. 'session 1').",
                        "data": None,
                        "context": _context_from_state(last_spec, last_session_range),
                    }
                resolved = resolve_relative_session(df, last_spec, cue)
                if "error" in resolved:
                    return {
                        "type": "error",
                        "answer": resolved["error"],
                        "data": None,
                        "context": _context_from_state(last_spec, last_session_range),
                    }
                sessions = [resolved["session"]]

            if len(sessions) >= 2:
                base = last_spec.model_copy(deep=True)
                base.session = sessions[0]
                cmp_out = compare_two_sessions(df, base, sessions[1])
            else:
                cmp_out = compare_two_sessions(df, last_spec, sessions[0])

            return {
                "type": "compare",
                "answer": narrate_session_comparison(cmp_out),
                "data": {"compare": cmp_out},
                "context": _context_from_state(last_spec, last_session_range),
            }

    # ---- LLM → QuerySpec ----
    llm_error = None
    try:
        spec = llm_question_to_query(question)
    except Exception as e:
        llm_error = e
        spec = None

    if spec is None:
        try:
            spec = deterministic_question_to_query(question)
        except Exception:
            return {
                "type": "error",
                "answer": f"LLM request failed: {llm_error}",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

    try:
        spec = apply_followup_context(spec, question, last_spec)

        # Resolve relative session cues like "next/previous/first/latest session"
        cue = detect_relative_session_cue(question)
        if cue is not None:
            if last_spec is None:
                raise ValueError("No prior session in context to resolve a relative session.")
            resolved = resolve_relative_session(df, last_spec, cue)
            if "error" in resolved:
                raise ValueError(resolved["error"])
            spec.session = resolved["session"]
            # Session-based query: clear date range
            spec.date_start = "__MISSING__"
            spec.date_end = "__MISSING__"

        if spec.metric not in ALLOWED_METRICS and spec.metric != "__MISSING__":
            raise ValueError(f"Metric '{spec.metric}' not allowed.")
        if spec.game is not None and spec.game not in ALLOWED_GAMES:
            raise ValueError(f"Game '{spec.game}' not allowed. Must be one of {ALLOWED_GAMES}.")
        if spec.session is not None and spec.session != "__MULTI__":
            if re.match(r"^session_\d+$", str(spec.session)) is None:
                raise ValueError(f"Session '{spec.session}' not allowed. Must match 'session_<number>'.")

    except (ValidationError, ValueError) as e:
        return {
            "type": "error",
            "answer": f"Model output failed strict validation: {e}",
            "data": None,
            "context": _context_from_state(last_spec, last_session_range),
        }
    except Exception as e:
        return {
            "type": "error",
            "answer": f"Unexpected error: {e}",
            "data": None,
            "context": _context_from_state(last_spec, last_session_range),
        }

    sessions_in_q = extract_sessions_from_text(question)
    explicit_session = question_mentions_session(question)
    explicit_dates = question_mentions_dates(question)

    # ---- SESSION RANGE MODE (single prompt) ----
    if len(sessions_in_q) >= 2 and _is_session_range_question(question):
        if spec.metric == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a metric for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.patient == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a patient for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.game is None:
            return {
                "type": "error",
                "answer": "Please specify the game for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

        session_start, session_end = sessions_in_q[0], sessions_in_q[1]
        results = run_session_range(df, spec, session_start, session_end)
        if len(results) == 1 and "error" in results[0]:
            return {
                "type": "error",
                "answer": results[0].get("error", "No results found."),
                "data": {"spec": spec.model_dump(), "results": results},
                "context": _context_from_state(last_spec, last_session_range),
            }

        summary = summarize_session_range(
            results,
            metric_name=spec.metric,
            session_start=session_start,
            session_end=session_end,
        )
        answer = narrate_session_range(summary, spec, session_start, session_end)
        last_spec = spec
        last_session_range = (session_start, session_end)
        return {
            "type": "session_range",
            "answer": answer,
            "data": {"spec": spec.model_dump(), "results": results, "summary": summary},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- SESSION RANGE FOLLOW-UP (re-use last range) ----
    if last_session_range and not sessions_in_q and not explicit_session and not explicit_dates:
        if spec.metric == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a metric for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.patient == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a patient for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.game is None:
            return {
                "type": "error",
                "answer": "Please specify the game for this session range.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

        session_start, session_end = last_session_range
        results = run_session_range(df, spec, session_start, session_end)
        if len(results) == 1 and "error" in results[0]:
            return {
                "type": "error",
                "answer": results[0].get("error", "No results found."),
                "data": {"spec": spec.model_dump(), "results": results},
                "context": _context_from_state(last_spec, last_session_range),
            }

        summary = summarize_session_range(
            results,
            metric_name=spec.metric,
            session_start=session_start,
            session_end=session_end,
        )
        answer = narrate_session_range(summary, spec, session_start, session_end)
        last_spec = spec
        return {
            "type": "session_range",
            "answer": answer,
            "data": {"spec": spec.model_dump(), "results": results, "summary": summary},
            "context": _context_from_state(last_spec, last_session_range),
        }

    if explicit_session or explicit_dates:
        last_session_range = None

    # ---- SESSION COMPARISON MODE (single prompt) ----
    if ("compare" in ql or "differ" in ql or "difference" in ql):
        if spec.metric == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a metric to compare.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.patient == "__MISSING__":
            return {
                "type": "error",
                "answer": "Please specify a patient to compare.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }
        if spec.game is None:
            return {
                "type": "error",
                "answer": "Please specify the game to compare.",
                "data": None,
                "context": _context_from_state(last_spec, last_session_range),
            }

        if len(sessions_in_q) >= 2:
            base = spec.model_copy(deep=True)
            base.session = sessions_in_q[0]
            cmp_out = compare_two_sessions(df, base, sessions_in_q[1])
        else:
            cue = detect_relative_session_cue(question)
            if len(sessions_in_q) == 1 and cue is not None:
                base = spec.model_copy(deep=True)
                base.session = sessions_in_q[0]
                resolved = resolve_relative_session(df, base, cue)
                if "error" in resolved:
                    return {
                        "type": "error",
                        "answer": resolved["error"],
                        "data": None,
                        "context": _context_from_state(last_spec, last_session_range),
                    }
                cmp_out = compare_two_sessions(df, base, resolved["session"])
            else:
                return {
                    "type": "error",
                    "answer": "Please mention the session number to compare.",
                    "data": None,
                    "context": _context_from_state(last_spec, last_session_range),
                }

        last_spec = spec
        last_session_range = None
        return {
            "type": "compare",
            "answer": narrate_session_comparison(cmp_out),
            "data": {"spec": spec.model_dump(), "compare": cmp_out},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- BASE QUERY ----
    results = run_query(df, spec)
    if len(results) == 1 and "error" in results[0]:
        return {
            "type": "error",
            "answer": results[0].get("error", "No results found."),
            "data": {"spec": spec.model_dump(), "results": results},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- POINT QUERY MODE ----
    if is_point_query(spec, results):
        point = format_point_result(results, spec.metric)
        answer = narrate_point(point, spec.metric, spec.patient)
        last_spec = spec
        return {
            "type": "point",
            "answer": answer,
            "data": {"spec": spec.model_dump(), "results": results, "point": point},
            "context": _context_from_state(last_spec, last_session_range),
        }

    # ---- TIME SERIES MODE ----
    summary = summarize_timeseries(
        results,
        metric_name=spec.metric,
        requested_start=spec.date_start,
    )
    answer = narrate_timeseries(summary, spec)
    if "error" not in summary:
        last_spec = spec
    return {
        "type": "timeseries",
        "answer": answer,
        "data": {"spec": spec.model_dump(), "results": results, "summary": summary},
        "context": _context_from_state(last_spec, last_session_range),
    }


def _should_code_fallback(error_answer: str) -> bool:
    if not ENABLE_CODE_FALLBACK:
        return False
    lower = error_answer.strip().lower()
    if lower.startswith("please specify"):
        return False
    if lower.startswith("llm request failed"):
        return False
    if "not allowed" in lower or "metric '" in lower:
        return False
    if "context cleared" in lower:
        return False
    return True


def process_question_with_fallback(
    question: str, df, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    resp = process_question(question, df, context)
    if resp.get("type") != "error":
        return resp

    error_answer = resp.get("answer", "")
    if not _should_code_fallback(error_answer):
        return resp

    fallback_context = {
        "deterministic_error": error_answer,
        "context": resp.get("context"),
    }

    try:
        result = run_code_fallback(question, CSV_PATH, fallback_context)
    except OpenAIFallbackError as e:
        resp["answer"] = f"{error_answer} (code fallback failed: {e})"
        return resp
    except Exception as e:
        resp["answer"] = f"{error_answer} (code fallback error: {e})"
        return resp

    answer = result.get("answer") or "Fallback completed."
    return {
        "type": "code_fallback",
        "answer": answer,
        "data": {
            "execution_path": "code_fallback",
            "fallback_reason": error_answer,
            "result": result,
        },
        "context": resp.get("context"),
    }
