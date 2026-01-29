# tests/test_chatbot_flow.py
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import llm_client
from config import ALLOWED_GAMES, ALLOWED_METRICS, ALLOWED_SESSIONS, CSV_PATH, RESET_COMMANDS
from context import (
    apply_followup_context,
    question_mentions_dates,
    question_mentions_session,
    question_mentions_patient,
    question_mentions_game,
    extract_metric_from_text,
)
from date_io import load_data
from narration import (
    narrate_point,
    narrate_session_comparison,
    narrate_session_range,
    narrate_timeseries,
)
from query_engine import (
    compare_two_sessions,
    detect_relative_session_cue,
    extract_sessions_from_text,
    resolve_relative_session,
    run_query,
    run_session_range,
)
from summarizer import (
    format_point_result,
    is_point_query,
    summarize_session_range,
    summarize_timeseries,
)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def install_llm_stub(monkeypatch: pytest.MonkeyPatch, mapping: dict) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, **kwargs):
            question = json["messages"][-1]["content"]
            if question not in mapping:
                raise AssertionError(f"Unexpected question for LLM stub: {question}")
            return FakeResponse(mapping[question])

    monkeypatch.setattr(llm_client.httpx, "Client", lambda *args, **kwargs: FakeClient())


def make_llm_json(**overrides) -> str:
    base = {
        "action": "get_metric_timeseries",
        "patient_id": "__MISSING__",
        "metric": "__MISSING__",
        "date_start": "__MISSING__",
        "date_end": "__MISSING__",
        "game": None,
        "session": None,
        "return_columns": ["date", "patient_id", "metric_value"],
    }
    base.update(overrides)
    return json.dumps(base)


class ChatbotHarness:
    def __init__(self, df):
        self.df = df
        self.last_spec = None
        self.last_session_range = None

    def ask(self, question: str):
        ql = question.strip().lower()

        if ql in RESET_COMMANDS:
            self.last_spec = None
            self.last_session_range = None
            return {"type": "reset"}

        def is_session_range_question(text: str) -> bool:
            t = text.lower()
            return ("from session" in t and "to session" in t) or ("between session" in t and "and session" in t)

        if ("differ" in ql or "difference" in ql or "compare" in ql) and self.last_spec is not None:
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
                        return {"type": "error", "error": "Please mention the session number to compare."}
                    resolved = resolve_relative_session(self.df, self.last_spec, cue)
                    if "error" in resolved:
                        return {"type": "error", "error": resolved["error"]}
                    sessions = [resolved["session"]]
                if len(sessions) >= 2:
                    base = self.last_spec.model_copy(deep=True)
                    base.session = sessions[0]
                    cmp_out = compare_two_sessions(self.df, base, sessions[1])
                else:
                    cmp_out = compare_two_sessions(self.df, self.last_spec, sessions[0])
                return {
                    "type": "compare",
                    "cmp": cmp_out,
                    "answer": narrate_session_comparison(cmp_out),
                }

        spec = llm_client.llm_question_to_query(question)
        spec = apply_followup_context(spec, question, self.last_spec)

        if spec.metric not in ALLOWED_METRICS and spec.metric != "__MISSING__":
            raise ValueError(f"Metric '{spec.metric}' not allowed.")
        if spec.game is not None and spec.game not in ALLOWED_GAMES:
            raise ValueError(f"Game '{spec.game}' not allowed. Must be one of {ALLOWED_GAMES}.")
        if spec.session is not None and spec.session != "__MULTI__" and spec.session not in ALLOWED_SESSIONS:
            raise ValueError(f"Session '{spec.session}' not allowed. Must be one of {ALLOWED_SESSIONS}.")

        sessions_in_q = extract_sessions_from_text(question)

        if len(sessions_in_q) >= 2 and is_session_range_question(question):
            if spec.metric == "__MISSING__":
                raise ValueError("Please specify a metric for this session range.")
            if spec.patient_id == "__MISSING__":
                raise ValueError("Please specify a patient for this session range.")
            if spec.game is None:
                raise ValueError("Please specify the game for this session range.")

            session_start = sessions_in_q[0]
            session_end = sessions_in_q[1]
            results = run_session_range(self.df, spec, session_start, session_end)
            if len(results) == 1 and "error" in results[0]:
                return {"type": "error", "error": results[0]["error"], "spec": spec}

            summary = summarize_session_range(results, spec.metric, session_start, session_end)
            answer = narrate_session_range(summary, spec, session_start, session_end)
            if "error" not in summary:
                self.last_spec = spec
                self.last_session_range = (session_start, session_end)
            return {
                "type": "session_range",
                "spec": spec,
                "answer": answer,
                "results": results,
                "summary": summary,
            }

        explicit_dates = question_mentions_dates(question)
        explicit_session = question_mentions_session(question)

        if self.last_session_range and not sessions_in_q and not explicit_dates and not explicit_session:
            if spec.metric == "__MISSING__":
                raise ValueError("Please specify a metric for this session range.")
            if spec.patient_id == "__MISSING__":
                raise ValueError("Please specify a patient for this session range.")
            if spec.game is None:
                raise ValueError("Please specify the game for this session range.")

            session_start, session_end = self.last_session_range
            results = run_session_range(self.df, spec, session_start, session_end)
            if len(results) == 1 and "error" in results[0]:
                return {"type": "error", "error": results[0]["error"], "spec": spec}

            summary = summarize_session_range(results, spec.metric, session_start, session_end)
            answer = narrate_session_range(summary, spec, session_start, session_end)
            if "error" not in summary:
                self.last_spec = spec
            return {
                "type": "session_range",
                "spec": spec,
                "answer": answer,
                "results": results,
                "summary": summary,
            }

        if explicit_dates or explicit_session:
            self.last_session_range = None

        if ("compare" in ql or "differ" in ql or "difference" in ql):
            if spec.metric == "__MISSING__":
                raise ValueError("Please specify a metric to compare.")
            if spec.patient_id == "__MISSING__":
                raise ValueError("Please specify a patient to compare.")
            if spec.game is None:
                raise ValueError("Please specify the game to compare.")

            if len(sessions_in_q) >= 2:
                base = spec.model_copy(deep=True)
                base.session = sessions_in_q[0]
                cmp_out = compare_two_sessions(self.df, base, sessions_in_q[1])
            else:
                cue = detect_relative_session_cue(question)
                if len(sessions_in_q) == 1 and cue is not None:
                    base = spec.model_copy(deep=True)
                    base.session = sessions_in_q[0]
                    resolved = resolve_relative_session(self.df, base, cue)
                    if "error" in resolved:
                        return {"type": "error", "error": resolved["error"]}
                    cmp_out = compare_two_sessions(self.df, base, resolved["session"])
                else:
                    return {"type": "error", "error": "Please mention the session number to compare."}
            self.last_spec = spec
            self.last_session_range = None
            return {
                "type": "compare",
                "cmp": cmp_out,
                "answer": narrate_session_comparison(cmp_out),
            }

        results = run_query(self.df, spec)
        if len(results) == 1 and "error" in results[0]:
            return {"type": "error", "error": results[0]["error"], "spec": spec}

        if is_point_query(spec, results):
            point = format_point_result(results, spec.metric)
            answer = narrate_point(point, spec.metric, spec.patient_id)
            self.last_spec = spec
            return {
                "type": "point",
                "spec": spec,
                "answer": answer,
                "results": results,
                "point": point,
            }

        summary = summarize_timeseries(results, metric_name=spec.metric, requested_start=spec.date_start)
        answer = narrate_timeseries(summary, spec)
        if "error" not in summary:
            self.last_spec = spec
        return {
            "type": "timeseries",
            "spec": spec,
            "answer": answer,
            "results": results,
            "summary": summary,
        }


def test_expected_conversation_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    df = load_data(CSV_PATH)

    q1 = "how has patient 45 m range of motion changed from 10/3/22 to 25/3/22 in game0?"
    q2 = "how has patient 45 m rom changed from 10/3/22 to 25/3/22 in game0?"
    q3 = "how has patient 45 m range_of_motion changed from 10/3/22 to 25/3/22 in game0?"
    q4 = "what about in game 1?"
    q5 = "how about from 24/2/22 to 10/3/22"
    q6 = "how about from 24/2/22 to 10/3/22 in game 1?"
    q7 = "and their force?"
    q8 = "tell me about patient 45m force in session 4 of game 2"
    q11 = "how does that differ from session 5?"
    q12 = "and how does that differ from session 1?"
    q13 = "compare this to their first session of game 2?"
    q14 = "compare this to the previous session?"
    q15 = "compare this to the next session?"
    q16 = "compare this to the latest session?"
    q17 = "how has patient 45 m area changed from 10/3/22 to 25/3/22 (no game)"
    q18 = "how did patient 45m force change from session 1 to session 7 of game 2"
    q19 = "compare patient 45 m force in session 1 and session 7 of game 2"
    q20 = "tell me about patient 45m sparc in game0 from 10/3/22 to 25/3/22"
    q21 = "what about his rom?"
    q22 = "how does it change from session 1 to session 6?"
    q23 = "compare session 6 to the latest session for sparc in game 2"
    q24 = "compare session 6 to the first session for sparc in game 2"

    mapping = {
        q1: make_llm_json(patient_id="45_M", metric="range_of_motion", game="game0"),
        q2: make_llm_json(patient_id="45_M", metric="rom", game="game0"),
        q3: make_llm_json(patient_id="45_M", metric="range_of_motion", game="game0"),
        q4: make_llm_json(game="game1"),
        q5: make_llm_json(),
        q6: make_llm_json(game="game1"),
        q7: make_llm_json(metric="force"),
        q8: make_llm_json(patient_id="45_M", metric="force", game="game2", session="session_4"),
        q17: make_llm_json(patient_id="45_M", metric="area"),
        q18: make_llm_json(patient_id="45_M", metric="force", game="game2", session="__MULTI__"),
        q19: make_llm_json(patient_id="45_M", metric="force", game="game2", session="__MULTI__"),
        q20: make_llm_json(patient_id="45_M", metric="average_sparc", game="game0"),
        q21: make_llm_json(metric="rom"),
        q22: make_llm_json(session="session_1__session_6"),
        q23: make_llm_json(patient_id="45_M", metric="average_sparc", game="game2", session="session_6"),
        q24: make_llm_json(patient_id="45_M", metric="average_sparc", game="game2", session="session_6"),
    }

    install_llm_stub(monkeypatch, mapping)

    bot = ChatbotHarness(df)

    out1 = bot.ask(q1)
    assert out1["spec"].metric == "area"
    assert out1["spec"].game == "game0"
    assert out1["spec"].date_start == "2022-03-10"
    assert out1["spec"].date_end == "2022-03-25"
    assert out1["answer"] == (
        "I found patient records from 2022-03-10 to 2022-03-24. "
        "Over this period, the average range of motion increased by 0.0149, "
        "which corresponds to a 42.12% change from the baseline. "
        "This change suggests increased range of motion. "
        "Values generally improved from session to session."
    )
    per_date = out1["summary"]["per_date_summary"]
    assert per_date == [
        {
            "date": "2022-03-10",
            "n_sessions": 1,
            "mean_metric_value": pytest.approx(0.0353219818056475),
            "min_metric_value": pytest.approx(0.0353219818056475),
            "max_metric_value": pytest.approx(0.0353219818056475),
        },
        {
            "date": "2022-03-23",
            "n_sessions": 2,
            "mean_metric_value": pytest.approx(0.0360055216147092),
            "min_metric_value": pytest.approx(0.0198877520377781),
            "max_metric_value": pytest.approx(0.0521232911916403),
        },
        {
            "date": "2022-03-24",
            "n_sessions": 1,
            "mean_metric_value": pytest.approx(0.0501986365898053),
            "min_metric_value": pytest.approx(0.0501986365898053),
            "max_metric_value": pytest.approx(0.0501986365898053),
        },
    ]
    assert out1["results"] == [
        {
            "date": "2022-03-10",
            "patient_id": "45_M",
            "metric_value": pytest.approx(0.0353219818056475),
            "game": "game0",
            "session": "session_1",
        },
        {
            "date": "2022-03-23",
            "patient_id": "45_M",
            "metric_value": pytest.approx(0.0521232911916403),
            "game": "game0",
            "session": "session_2",
        },
        {
            "date": "2022-03-23",
            "patient_id": "45_M",
            "metric_value": pytest.approx(0.0198877520377781),
            "game": "game0",
            "session": "session_3",
        },
        {
            "date": "2022-03-24",
            "patient_id": "45_M",
            "metric_value": pytest.approx(0.0501986365898053),
            "game": "game0",
            "session": "session_4",
        },
    ]

    out2 = bot.ask(q2)
    assert out2["spec"].metric == "area"
    assert "This change suggests increased range of motion." in out2["answer"]
    assert "Values generally improved from session to session." in out2["answer"]

    out3 = bot.ask(q3)
    assert out3["spec"].metric == "area"
    assert "This change suggests increased range of motion." in out3["answer"]
    assert "Values generally improved from session to session." in out3["answer"]

    out4 = bot.ask(q4)
    assert out4["type"] == "point"
    assert out4["answer"] == (
        "For patient 45_M, the range of motion (area) value is 0.059413 "
        "in game1, session_1 on 2022-03-10."
    )
    assert out4["results"] == [
        {
            "date": "2022-03-10",
            "patient_id": "45_M",
            "metric_value": pytest.approx(0.0594131464752734),
            "game": "game1",
            "session": "session_1",
        }
    ]

    with pytest.raises(ValueError, match="For date-range queries, please specify the game"):
        bot.ask(q5)

    out6 = bot.ask(q6)
    assert out6["answer"] == (
        "I found patient records from 2022-02-25 to 2022-03-10. "
        "Over this period, the average range of motion increased by 0.0046, "
        "which corresponds to a 8.43% change from the baseline. "
        "No data on requested start date 2022-02-24; using first available date "
        "2022-02-25 as baseline. "
        "This change is consistent with increased range of motion. "
        "Values generally improved from session to session."
    )

    out7 = bot.ask(q7)
    assert out7["spec"].metric == "f_patient"
    assert out7["answer"] == (
        "I found patient records from 2022-02-25 to 2022-03-10. "
        "Over this period, the average applied force remained stable by 0.0000. "
        "No data on requested start date 2022-02-24; using first available date "
        "2022-02-25 as baseline. "
        "This change is consistent with similar strength over this period. "
        "Values stayed roughly stable between sessions."
    )

    out8 = bot.ask(q8)
    assert out8["type"] == "error"
    assert out8["error"] == "No matching rows found in uploaded CSV for that query."

    assert bot.ask("reset")["type"] == "reset"
    assert bot.last_spec is None

    out10 = bot.ask(q8)
    assert out10["type"] == "point"
    assert out10["answer"] == (
        "For patient 45_M, the total force value is -5.344677 in game2, "
        "session_4 on 2022-03-24."
    )

    out11 = bot.ask(q11)
    assert out11["answer"] == (
        "For patient 45_M in game2, comparing session_4 to session_5, "
        "the average applied force decreased by 7.3534 (from -5.3447 to -12.6980). "
        "The difference (earlier - later) is 7.3534. "
        "This corresponds to a 137.58% change relative to the earlier session. "
        "This change suggests reduced strength output."
    )

    out12 = bot.ask(q12)
    assert out12["answer"] == (
        "For patient 45_M in game2, comparing session_1 to session_4, "
        "the average applied force increased by 7.6698 (from -13.0145 to -5.3447). "
        "The difference (earlier - later) is -7.6698. "
        "This corresponds to a 58.93% change relative to the earlier session. "
        "This change suggests increased strength output."
    )

    bot.ask("reset")
    with pytest.raises(ValueError, match="For date-range queries, please specify the game"):
        bot.ask(q17)

    # Relative session comparisons (base: session_4 in game2)
    bot.ask("reset")
    out_base = bot.ask(q8)
    assert out_base["type"] == "point"

    out_first = bot.ask(q13)
    assert out_first["answer"] == (
        "For patient 45_M in game2, comparing session_1 to session_4, "
        "the average applied force increased by 7.6698 (from -13.0145 to -5.3447). "
        "The difference (earlier - later) is -7.6698. "
        "This corresponds to a 58.93% change relative to the earlier session. "
        "This change suggests increased strength output."
    )

    out_prev = bot.ask(q14)
    assert out_prev["answer"] == (
        "For patient 45_M in game2, comparing session_1 to session_4, "
        "the average applied force increased by 7.6698 (from -13.0145 to -5.3447). "
        "The difference (earlier - later) is -7.6698. "
        "This corresponds to a 58.93% change relative to the earlier session. "
        "This change suggests increased strength output."
    )

    out_next = bot.ask(q15)
    assert out_next["answer"] == (
        "For patient 45_M in game2, comparing session_4 to session_5, "
        "the average applied force decreased by 7.3534 (from -5.3447 to -12.6980). "
        "The difference (earlier - later) is 7.3534. "
        "This corresponds to a 137.58% change relative to the earlier session. "
        "This change suggests reduced strength output."
    )

    out_latest = bot.ask(q16)
    assert out_latest["answer"] == (
        "For patient 45_M in game2, comparing session_4 to session_7, "
        "the average applied force decreased by 2.7810 (from -5.3447 to -8.1257). "
        "The difference (earlier - later) is 2.7810. "
        "This corresponds to a 52.03% change relative to the earlier session. "
        "This change suggests reduced strength output."
    )

    out_range = bot.ask(q18)
    assert out_range["type"] == "session_range"
    assert out_range["answer"] == (
        "I found patient records from session_1 to session_7. "
        "Over this session range, the average applied force increased by 4.8887, "
        "which corresponds to a 37.56% change from the baseline. "
        "This change suggests increased strength output. "
        "Overall, the trend shows fluctuations with rises and drops between sessions, "
        "but it increased overall from the first to the last session."
    )

    out_range_follow = bot.ask(q21)
    assert out_range_follow["type"] == "session_range"
    assert out_range_follow["answer"] == (
        "I found patient records from session_1 to session_7. "
        "Over this session range, the average range of motion increased by 0.0049, "
        "which corresponds to a 11.95% change from the baseline. "
        "This change is consistent with increased range of motion. "
        "Overall, the trend shows fluctuations with rises and drops between sessions, "
        "but it increased overall from the first to the last session."
    )

    out_compare_single = bot.ask(q19)
    assert out_compare_single["answer"] == (
        "For patient 45_M in game2, comparing session_1 to session_7, "
        "the average applied force increased by 4.8887 (from -13.0145 to -8.1257). "
        "The difference (earlier - later) is -4.8887. "
        "This corresponds to a 37.56% change relative to the earlier session. "
        "This change suggests increased strength output."
    )

    out_range_follow2 = bot.ask(q22)
    assert out_range_follow2["type"] == "session_range"
    assert out_range_follow2["answer"] == (
        "I found patient records from session_1 to session_6. "
        "Over this session range, the average applied force increased by 2.5721, "
        "which corresponds to a 19.76% change from the baseline. "
        "This change suggests increased strength output. "
        "Overall, the trend shows fluctuations with rises and drops between sessions, "
        "but it increased overall from the first to the last session."
    )

    out_latest_sparc = bot.ask(q23)
    assert out_latest_sparc["type"] == "compare"
    assert out_latest_sparc["answer"] == (
        "For patient 45_M in game2, comparing session_6 to session_7, "
        "the average movement smoothness increased by 9.9617 (from -28.2029 to -18.2412). "
        "The difference (earlier - later) is -9.9617. "
        "This corresponds to a 35.32% change relative to the earlier session. "
        "This change suggests smoother, more continuous, better-coordinated movement."
    )

    out_first_sparc = bot.ask(q24)
    assert out_first_sparc["type"] == "compare"
    assert out_first_sparc["answer"] == (
        "For patient 45_M in game2, comparing session_1 to session_6, "
        "the average movement smoothness increased by 6.1213 (from -34.3243 to -28.2029). "
        "The difference (earlier - later) is -6.1213. "
        "This corresponds to a 17.83% change relative to the earlier session. "
        "This change suggests smoother, more continuous, better-coordinated movement."
    )

    out_after_range = bot.ask(q20)
    assert out_after_range["type"] == "timeseries"
    assert out_after_range["answer"] == (
        "I found patient records from 2022-03-10 to 2022-03-24. "
        "Over this period, the average movement smoothness increased by 0.9132, "
        "which corresponds to a 36.54% change from the baseline. "
        "This change suggests smoother, more continuous, better-coordinated movement. "
        "Overall, the trend shows fluctuations with rises and drops between sessions, "
        "but it increased overall from the first to the last date."
    )
