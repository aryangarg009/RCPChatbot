# chatbot.py
import json
from typing import Optional

from pydantic import ValidationError

from config import (
    CSV_PATH,
    RESET_COMMANDS,
    ALLOWED_METRICS,
    ALLOWED_GAMES,
    ALLOWED_SESSIONS,
    ENABLE_CODE_FALLBACK,
)
from schema import QuerySpec
from date_io import load_data, extract_dates_from_text
from llm_client import llm_question_to_query
from context import (
    apply_followup_context,
    is_metric_definition_question,
    extract_metric_or_alias_from_definition_question,
    question_mentions_dates,
    question_mentions_session,
    question_mentions_patient,
    question_mentions_game,
    extract_metric_from_text,
)
from query_engine import (
    run_query,
    extract_sessions_from_text,
    detect_relative_session_cue,
    resolve_relative_session,
    compare_two_sessions,
    run_session_range,
)
from summarizer import summarize_timeseries, summarize_session_range, is_point_query, format_point_result
from narration import narrate_timeseries, narrate_point, narrate_session_comparison, narrate_session_range
from metrics import METRIC_EXPLANATIONS
from openai_fallback import OpenAIFallbackError, run_code_fallback

def main():
    print("Loading CSV...")
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print("Ask a question. Type 'quit' to exit.\n")

    last_spec: Optional[QuerySpec] = None
    last_session_range: Optional[tuple[str, str]] = None

    def _should_code_fallback(error_text: str) -> bool:
        if not ENABLE_CODE_FALLBACK:
            return False
        lower = error_text.strip().lower()
        if lower.startswith("please specify"):
            return False
        if "context cleared" in lower:
            return False
        return True

    def _try_code_fallback(question: str, reason: str) -> bool:
        if not _should_code_fallback(reason):
            return False
        context = {
            "deterministic_error": reason,
            "last_spec": last_spec.model_dump() if last_spec is not None else None,
            "last_session_range": list(last_session_range) if last_session_range else None,
        }
        try:
            result = run_code_fallback(question, CSV_PATH, context)
        except OpenAIFallbackError as e:
            print(f"\n[ERROR] Code fallback failed: {e}\n")
            return False
        except Exception as e:
            print(f"\n[ERROR] Code fallback error: {e}\n")
            return False

        print("\n[CODE FALLBACK]")
        print(result.get("answer", "Fallback completed."))
        if "data" in result:
            print("\nData:")
            print(json.dumps(result["data"], indent=2))
        if "warnings" in result and result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"- {w}")
        print("")
        return True

    while True:
        q = input("You: ").strip()
        ql = q.lower()

        def is_session_range_question(text: str) -> bool:
            t = text.lower()
            return ("from session" in t and "to session" in t) or ("between session" in t and "and session" in t)

        if ql in {"quit", "exit"}:
            break

        if ql in RESET_COMMANDS:
            last_spec = None
            last_session_range = None
            print("\n[OK] Context cleared. Ask a new question with patient/metric/date again.\n")
            continue

        # ---- METRIC DEFINITION MODE ----
        if is_metric_definition_question(q):
            metric = extract_metric_or_alias_from_definition_question(q)

            if metric is None:
                print(
                    "\nI’m not sure which metric you mean. "
                    "Try: 'what is sparc?' or 'what does efficiency mean?'\n"
                )
                continue

            explanation = METRIC_EXPLANATIONS.get(metric)
            if explanation is None:
                print(
                    f"\nI don’t have an explanation written yet for '{metric}', "
                    "but I can add one.\n"
                )
                continue

            print(f"\n{explanation}\n")
            continue

        # ---- SESSION COMPARISON MODE (follow-up) ----
        if ("differ" in ql or "difference" in ql or "compare" in ql) and last_spec is not None:
            # If user explicitly mentions patient/metric/game, treat as standalone compare
            if (
                question_mentions_patient(q)
                or question_mentions_game(q)
                or extract_metric_from_text(q) is not None
            ):
                pass
            else:
                sessions = extract_sessions_from_text(q)

                if not sessions:
                    cue = detect_relative_session_cue(q)
                    if cue is None:
                        print("\n[BLOCKED] Please mention the session number to compare (e.g. 'session 1').\n")
                        continue
                    resolved = resolve_relative_session(df, last_spec, cue)
                    if "error" in resolved:
                        print(f"\n[BLOCKED] {resolved['error']}\n")
                        continue
                    sessions = [resolved["session"]]

                if len(sessions) >= 2:
                    base = last_spec.model_copy(deep=True)
                    base.session = sessions[0]
                    cmp_out = compare_two_sessions(df, base, sessions[1])
                else:
                    cmp_out = compare_two_sessions(df, last_spec, sessions[0])

                print("\nAnswer:")
                if "error" in cmp_out:
                    print(cmp_out["error"])
                    print("")
                    _try_code_fallback(q, cmp_out["error"])
                    continue

                print(narrate_session_comparison(cmp_out))

                print(f"\nRaw rows (earlier session: {cmp_out['session_earlier']}):")
                for r in cmp_out["rows_earlier"]:
                    print(r)
                print(f"\nRaw rows (later session: {cmp_out['session_later']}):")
                for r in cmp_out["rows_later"]:
                    print(r)
                print("")
                continue

        try:
            spec = llm_question_to_query(q)

            # Apply follow-up context safely
            spec = apply_followup_context(spec, q, last_spec)

            # SAFETY: if user typed dates in this question, do not allow context to overwrite them
            if extract_dates_from_text(q):
                pass  # dates are explicit; trust llm_client's override

            # Re-run strict checks after context fill
            if spec.metric not in ALLOWED_METRICS and spec.metric != "__MISSING__":
                raise ValueError(f"Metric '{spec.metric}' not allowed.")

            if spec.game is not None and spec.game not in ALLOWED_GAMES:
                raise ValueError(f"Game '{spec.game}' not allowed. Must be one of {ALLOWED_GAMES}.")

            if spec.session is not None and spec.session != "__MULTI__" and spec.session not in ALLOWED_SESSIONS:
                raise ValueError(f"Session '{spec.session}' not allowed. Must be one of {ALLOWED_SESSIONS}.")

            print("\n[QuerySpec used]:", spec.model_dump())

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print("\n[BLOCKED] Model output failed strict validation.")
            print(f"Reason: {e}\n")
            _try_code_fallback(q, f"Model output failed strict validation: {e}")
            continue
        except Exception as e:
            print(f"\n[ERROR] LLM request failed: {e}\n")
            _try_code_fallback(q, f"LLM request failed: {e}")
            continue

        sessions_in_q = extract_sessions_from_text(q)
        explicit_session = question_mentions_session(q)
        explicit_dates = question_mentions_dates(q)

        # ---- SESSION RANGE MODE (single prompt) ----
        if len(sessions_in_q) >= 2 and is_session_range_question(q):
            if spec.metric == "__MISSING__":
                print("\n[BLOCKED] Please specify a metric for this session range.\n")
                continue
            if spec.patient_id == "__MISSING__":
                print("\n[BLOCKED] Please specify a patient for this session range.\n")
                continue
            if spec.game is None:
                print("\n[BLOCKED] Please specify the game for this session range.\n")
                continue

            session_start = sessions_in_q[0]
            session_end = sessions_in_q[1]
            results = run_session_range(df, spec, session_start, session_end)

            if len(results) == 1 and "error" in results[0]:
                print("\nAnswer:")
                print(results[0].get("error", "No results found."))
                print("")
                _try_code_fallback(q, results[0].get("error", "No results found."))
                continue

            summary = summarize_session_range(
                results,
                metric_name=spec.metric,
                session_start=session_start,
                session_end=session_end,
            )

            print("\nAnswer:")
            print(narrate_session_range(summary, spec, session_start, session_end))

            if "error" in summary:
                print("")
                continue

            print("Per-session:")
            for row in summary["per_session_summary"]:
                print(row)

            print("\nRaw rows (audit trail):")
            for item in results:
                print(item)
            print("")
            last_spec = spec
            last_session_range = (session_start, session_end)
            continue

        # ---- SESSION RANGE FOLLOW-UP (re-use last range) ----
        if last_session_range and not sessions_in_q and not explicit_session and not explicit_dates:
            if spec.metric == "__MISSING__":
                print("\n[BLOCKED] Please specify a metric for this session range.\n")
                continue
            if spec.patient_id == "__MISSING__":
                print("\n[BLOCKED] Please specify a patient for this session range.\n")
                continue
            if spec.game is None:
                print("\n[BLOCKED] Please specify the game for this session range.\n")
                continue

            session_start, session_end = last_session_range
            results = run_session_range(df, spec, session_start, session_end)

            if len(results) == 1 and "error" in results[0]:
                print("\nAnswer:")
                print(results[0].get("error", "No results found."))
                print("")
                _try_code_fallback(q, results[0].get("error", "No results found."))
                continue

            summary = summarize_session_range(
                results,
                metric_name=spec.metric,
                session_start=session_start,
                session_end=session_end,
            )

            print("\nAnswer:")
            print(narrate_session_range(summary, spec, session_start, session_end))

            if "error" in summary:
                print("")
                continue

            print("Per-session:")
            for row in summary["per_session_summary"]:
                print(row)

            print("\nRaw rows (audit trail):")
            for item in results:
                print(item)
            print("")
            last_spec = spec
            continue

        # ---- SESSION COMPARISON MODE (single prompt) ----
        if ("compare" in ql or "differ" in ql or "difference" in ql):
            if spec.metric == "__MISSING__":
                print("\n[BLOCKED] Please specify a metric to compare.\n")
                continue
            if spec.patient_id == "__MISSING__":
                print("\n[BLOCKED] Please specify a patient to compare.\n")
                continue
            if spec.game is None:
                print("\n[BLOCKED] Please specify the game to compare.\n")
                continue
            if len(sessions_in_q) >= 2:
                base = spec.model_copy(deep=True)
                base.session = sessions_in_q[0]
                cmp_out = compare_two_sessions(df, base, sessions_in_q[1])
            else:
                cue = detect_relative_session_cue(q)
                if len(sessions_in_q) == 1 and cue is not None:
                    base = spec.model_copy(deep=True)
                    base.session = sessions_in_q[0]
                    resolved = resolve_relative_session(df, base, cue)
                    if "error" in resolved:
                        print(f"\n[BLOCKED] {resolved['error']}\n")
                        continue
                    cmp_out = compare_two_sessions(df, base, resolved["session"])
                else:
                    # No explicit or resolvable sessions in a standalone compare
                    continue

            print("\nAnswer:")
            if "error" in cmp_out:
                print(cmp_out["error"])
                print("")
                _try_code_fallback(q, cmp_out["error"])
                continue

            print(narrate_session_comparison(cmp_out))

            print(f"\nRaw rows (earlier session: {cmp_out['session_earlier']}):")
            for r in cmp_out["rows_earlier"]:
                print(r)
            print(f"\nRaw rows (later session: {cmp_out['session_later']}):")
            for r in cmp_out["rows_later"]:
                print(r)
            print("")
            last_spec = spec
            last_session_range = None
            continue

        if explicit_session or explicit_dates:
            last_session_range = None

        results = run_query(df, spec)

        if len(results) == 1 and "error" in results[0]:
            print("\nAnswer:")
            print(results[0].get("error", "No results found."))
            print("")
            _try_code_fallback(q, results[0].get("error", "No results found."))
            continue

        # ---- POINT QUERY MODE ----
        if is_point_query(spec, results):
            point = format_point_result(results, spec.metric)

            print("\nAnswer:")
            print(narrate_point(point, spec.metric, spec.patient_id))

            print("\nRaw rows (audit trail):")
            for r in results:
                print(r)
            print("")
            last_spec = spec
            continue

        # ---- TIME SERIES MODE ----
        summary = summarize_timeseries(
            results,
            metric_name=spec.metric,
            requested_start=spec.date_start
        )

        # Natural-language answer (deterministic, no LLM)
        print("\nAnswer:")
        print(narrate_timeseries(summary, spec))

        if not (len(results) == 1 and "error" in results[0]):
            last_spec = spec

        if "error" in summary:
            print(summary["error"])
            print("\nRaw rows (audit trail):")
            for item in results:
                print(item)
            print("")
            continue

        print("Per-date:")
        for row in summary["per_date_summary"]:
            print(row)

        print("\nRaw rows (audit trail):")
        for item in results:
            print(item)
        print("")

if __name__ == "__main__":
    main()
