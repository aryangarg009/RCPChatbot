# tests/test_chatbot_flow.py
import json
import os
import sys

import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import llm_client
from config import RESET_COMMANDS
from context import (
    apply_followup_context,
    question_mentions_dates,
    question_mentions_session,
    question_mentions_patient,
    question_mentions_game,
    extract_metric_from_text,
)
from narration import narrate_point, narrate_session_comparison, narrate_session_range, narrate_timeseries
from query_engine import (
    compare_two_sessions,
    detect_relative_session_cue,
    extract_sessions_from_text,
    resolve_relative_session,
    run_query,
    run_session_range,
)
from summarizer import format_point_result, is_point_query, summarize_session_range, summarize_timeseries


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
        "patient": "__MISSING__",
        "metric": "__MISSING__",
        "date_start": "__MISSING__",
        "date_end": "__MISSING__",
        "game": None,
        "session": None,
        "return_columns": ["date", "patient", "metric_value"],
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

        sessions_in_q = extract_sessions_from_text(question)

        if len(sessions_in_q) >= 2 and is_session_range_question(question):
            if spec.metric == "__MISSING__":
                raise ValueError("Please specify a metric for this session range.")
            if spec.patient == "__MISSING__":
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
            if spec.patient == "__MISSING__":
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
            if spec.patient == "__MISSING__":
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
            answer = narrate_point(point, spec.metric, spec.patient)
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


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": ["46", "46", "46"],
            "gender": ["M", "M", "M"],
            "game": ["game0", "game0", "game0"],
            "session": ["session_1", "session_1", "session_2"],
            "date": pd.to_datetime(
                [
                    "2022-11-07T09:51:02.000",
                    "2022-11-07T10:09:27.000",
                    "2022-11-10T11:41:52.000",
                ]
            ),
            "timestampms": [90000, 120000, 60000],
            "avg_efficiency": [0.8, 0.9, 0.7],
            "area": [0.1, 0.12, 0.08],
            "average_sparc": [-2.0, -2.1, -2.2],
            "avg_f_patient": [-10.0, -12.0, -9.0],
        }
    )


def test_iso_datetime_dates_are_filtered_by_date_only(monkeypatch: pytest.MonkeyPatch, tiny_df) -> None:
    q = "how has patient 46 range of motion changed from 7th November 2022 to 7th November 2022 in game0?"
    mapping = {
        q: make_llm_json(patient="46", metric="range_of_motion", game="game0"),
    }
    install_llm_stub(monkeypatch, mapping)

    bot = ChatbotHarness(tiny_df)
    out = bot.ask(q)

    assert out["spec"].date_start == "2022-11-07"
    assert out["spec"].date_end == "2022-11-07"
    assert out["type"] == "timeseries"
    assert len(out["results"]) == 2


def test_duration_query_returns_readable_time(monkeypatch: pytest.MonkeyPatch, tiny_df) -> None:
    q = "how long was patient 46 session 1 in game0"
    mapping = {
        q: make_llm_json(patient="46", session="session_1", game="game0"),
    }
    install_llm_stub(monkeypatch, mapping)

    bot = ChatbotHarness(tiny_df)
    out = bot.ask(q)

    assert out["type"] == "point"
    assert "session duration" in out["answer"]
    assert "m" in out["answer"] or "s" in out["answer"]


def test_gender_question_is_detected(tiny_df) -> None:
    from chat_service import process_question

    out = process_question("what is the gender of patient 46?", tiny_df, context=None)
    assert out["type"] == "gender"
    assert out["answer"] == "Patient 46 is male."
