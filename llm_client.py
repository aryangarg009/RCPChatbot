# llm_client.py
import json
import os
import re
from typing import Optional

import httpx
from pydantic import ValidationError

import config as cfg
from config import (
    PARSER_BACKEND,
    OPENAI_API_BASE,
    OPENAI_MODEL,
    ALLOWED_METRICS, ALLOWED_GAMES, ALLOWED_SESSIONS
)
from schema import QuerySpec
from date_io import parse_date_to_iso, apply_open_ended_date_logic, extract_dates_from_text
from query_engine import normalize_session_string, detect_relative_session_cue, extract_sessions_from_text  # safe, no circular import
from context import normalize_metric_alias, extract_patient_from_text, is_duration_question, extract_metric_from_text

def extract_json_strict(text: str) -> str:
    """
    We only accept a JSON object.
    This rejects extra text before/after JSON.
    """
    text = text.strip()

    # If the model returned code fences, strip them
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    # Must be a single JSON object
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("Model did not return a single JSON object.")

    return text

def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return api_key

def normalize_llm_obj(obj: dict) -> dict:
    """
    Convert missing/None LLM outputs into our explicit __MISSING__ placeholders
    so QuerySpec validation never crashes before follow-up context can apply.
    """
    if "patient" not in obj and "patient_id" in obj:
        obj["patient"] = obj.get("patient_id")
    obj.setdefault("action", "get_metric_timeseries")
    obj.setdefault("patient", "__MISSING__")
    obj.setdefault("metric", "__MISSING__")
    obj.setdefault("date_start", "__MISSING__")
    obj.setdefault("date_end", "__MISSING__")
    obj.setdefault("game", None)
    obj.setdefault("session", None)
    obj.setdefault("return_columns", ["date", "patient", "metric_value"])

    if obj.get("patient") is None:
        obj["patient"] = "__MISSING__"
    if obj.get("metric") is None:
        obj["metric"] = "__MISSING__"
    if obj.get("date_start") is None:
        obj["date_start"] = "__MISSING__"
    if obj.get("date_end") is None:
        obj["date_end"] = "__MISSING__"
    if isinstance(obj.get("session"), list):
        obj["session"] = "__MULTI__"

    return obj

def _find_disallowed_metric_token(question: str) -> Optional[str]:
    """
    Detect explicit snake_case metric tokens that are NOT in ALLOWED_METRICS.
    This prevents the LLM from silently substituting a different metric.
    """
    tokens = re.findall(r"\b[a-zA-Z]+_[a-zA-Z0-9_]+\b", question)
    allowed_lower = {m.lower() for m in ALLOWED_METRICS}
    for tok in tokens:
        t = tok.lower()
        if re.match(r"^session_\d+$", t):
            continue
        if re.match(r"^game\d+$", t):
            continue
        if t in allowed_lower:
            continue
        return tok
    return None

def llm_question_to_query(question: str) -> QuerySpec:
    bad_token = _find_disallowed_metric_token(question)
    if bad_token is not None:
        raise ValueError(f"Metric '{bad_token}' not allowed.")

    system_prompt = f"""
You are a strict query generator for a medical data CSV.
You MUST output ONLY ONE valid JSON object and NOTHING ELSE.

Rules:
- Do NOT answer the question.
- Do NOT include any numbers from the dataset.
- Only generate JSON that matches this action: "get_metric_timeseries".
- metric must be EXACTLY one of: {ALLOWED_METRICS}
- If the user uses an alias, map it to a valid metric name. Examples:
  - "smoothness" or "sparc" -> "average_sparc"
  - "range of motion" or "rom" -> "area"
  - "efficiency" -> "avg_efficiency"
  - "force" or "strength" -> "avg_f_patient"
- "session duration" or "how long" -> "timestampms"
- patient must be the exact digits (e.g., "46") if mentioned.
- If session is null, date_start must be present. date_end may be "__MISSING__" for open-ended queries like "since <date>".
- If a session is specified and the question does not include dates, set date_start and date_end to "__MISSING__".
- If game/session not specified, set them to null.
- return_columns must be exactly: ["date","patient","metric_value"].
- If the question mentions a game like "game0", "game1", "game2", or "game3", set "game" to that exact string (case-sensitive). Otherwise set "game" to null.
- Do NOT guess the game. Only set it if explicitly mentioned in the user question.
- If the question mentions a session like "session_1", "session_2", etc., set "session" to that exact string (case-sensitive). Otherwise set "session" to null.
- Do NOT guess the session. Only set it if explicitly mentioned in the user question.
- If the question uses relative session language (next/previous/latest/first), set "session" to null (do NOT use __NEXT__/__PREVIOUS__/etc).
- If the question mentions MORE THAN ONE game, set "game" to "__MULTI__".
- If the question mentions MORE THAN ONE session, set "session" to "__MULTI__".

If the question is missing patient or metric, output:
{{"action":"get_metric_timeseries","patient":"__MISSING__","metric":"__MISSING__","date_start":"__MISSING__","date_end":"__MISSING__","game":null,"session":null,"return_columns":["date","patient","metric_value"]}}

If a session is explicitly specified in the question and dates are not mentioned,
it is allowed for date_start and date_end to be "__MISSING__".
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        if PARSER_BACKEND.lower() == "lmstudio":
            lmstudio_url = getattr(cfg, "LMSTUDIO_URL", None)
            lmstudio_model = getattr(cfg, "MODEL", None)
            if not lmstudio_url or not lmstudio_model:
                raise ValueError("LM Studio parser selected but LMSTUDIO_URL/MODEL not configured.")
            payload["model"] = lmstudio_model
            r = client.post(lmstudio_url, json=payload)
        else:
            headers = {
                "Authorization": f"Bearer {_get_openai_api_key()}",
                "Content-Type": "application/json",
            }
            r = client.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    json_text = extract_json_strict(content)
    obj = json.loads(json_text)
    obj = normalize_llm_obj(obj)

    # Validate schema (hard guardrail)
    spec = QuerySpec(**obj)

    extracted_patient = extract_patient_from_text(question)
    if extracted_patient is not None:
        spec.patient = extracted_patient

    if spec.metric == "__MISSING__" and is_duration_question(question):
        spec.metric = "timestampms"

    # Normalize session formats like "session 2" -> "session_2"
    if spec.session is not None and spec.session != "__MULTI__":
        # If the model crammed multiple sessions into one field, mark as MULTI
        session_hits = re.findall(r"session[_\s]*\d+", str(spec.session).lower())
        if len(session_hits) >= 2:
            spec.session = "__MULTI__"
        else:
            ns = normalize_session_string(spec.session)
            if ns is not None:
                spec.session = ns
            elif str(spec.session).upper() in {"__NEXT__", "__PREVIOUS__", "__FIRST__", "__LATEST__"}:
                spec.session = None

    # If the user used relative session language, defer resolution to follow-up logic
    if detect_relative_session_cue(question) is not None:
        spec.session = None

    # Normalize metric aliases (e.g., "range_of_motion" -> "area")
    spec.metric = normalize_metric_alias(spec.metric, question)
 
    # Deterministic date override
    spec = apply_open_ended_date_logic(spec, question)

    # BLOCK multi-game
    if spec.game == "__MULTI__":
        raise ValueError("Multiple games mentioned. Please specify only one game for now.")

    # Validate metric explicitly
    if spec.metric not in ALLOWED_METRICS and spec.metric != "__MISSING__":
        raise ValueError(f"Metric '{spec.metric}' not allowed.")

    # Validate game explicitly
    if spec.game is not None and spec.game not in ALLOWED_GAMES:
        raise ValueError(f"Game '{spec.game}' not allowed. Must be one of {ALLOWED_GAMES}.")

    # Validate session explicitly (format only)
    if spec.session is not None and spec.session != "__MULTI__":
        if re.match(r"^session_\d+$", str(spec.session)) is None:
            raise ValueError(f"Session '{spec.session}' not allowed. Must match 'session_<number>'.")

    # Normalize dates to ISO (ONLY if not already ISO)
    iso_pat = r"^\d{4}-\d{2}-\d{2}$"

    if spec.date_start != "__MISSING__" and not re.match(iso_pat, spec.date_start):
        spec.date_start = parse_date_to_iso(spec.date_start)

    if spec.date_end != "__MISSING__" and not re.match(iso_pat, spec.date_end):
        spec.date_end = parse_date_to_iso(spec.date_end)

    # Validate date ordering AFTER normalization
    if spec.date_start != "__MISSING__" and spec.date_end != "__MISSING__":
        if spec.date_start > spec.date_end:
            raise ValueError(
                f"Start date {spec.date_start} is after end date {spec.date_end}. "
                "Check D/M/Y vs M/D/Y."
            )
        
    # If no dates are mentioned and a session is present, require game too
    if spec.session is not None and spec.session != "__MULTI__" and len(extract_dates_from_text(question)) == 0:
        if spec.game is None:
            raise ValueError("Please specify the game for a session query (e.g., 'session 4 in game0').")
        
    # Enforce: date-range queries must specify game (to avoid cross-game mixing)
    has_dates = len(extract_dates_from_text(question)) > 0
    if spec.session is None and has_dates and spec.game is None:
        raise ValueError("For date-range queries, please specify the game (e.g., 'in game0 from 10/3/22 to 24/3/22').")
   
    return spec


def deterministic_question_to_query(question: str) -> QuerySpec:
    """
    Deterministically parse question into QuerySpec without LLM.
    Uses regex/date parsing and metric aliases only.
    """
    obj = {
        "action": "get_metric_timeseries",
        "patient": "__MISSING__",
        "metric": "__MISSING__",
        "date_start": "__MISSING__",
        "date_end": "__MISSING__",
        "game": None,
        "session": None,
        "return_columns": ["date", "patient", "metric_value"],
    }

    patient = extract_patient_from_text(question)
    if patient is not None:
        obj["patient"] = patient

    metric = extract_metric_from_text(question)
    if metric is not None:
        obj["metric"] = metric

    # Game parsing: accept "game0" or "game 0"
    games = re.findall(r"\bgame\s*\d+\b", question.lower())
    if len(games) >= 2:
        obj["game"] = "__MULTI__"
    elif len(games) == 1:
        obj["game"] = games[0].replace(" ", "")

    # Session parsing
    sessions = extract_sessions_from_text(question)
    if len(sessions) >= 2:
        obj["session"] = "__MULTI__"
    elif len(sessions) == 1:
        obj["session"] = sessions[0]

    # Relative session cues: defer resolution
    if detect_relative_session_cue(question) is not None:
        obj["session"] = None

    spec = QuerySpec(**obj)

    if spec.session is not None and spec.session != "__MULTI__":
        ns = normalize_session_string(spec.session)
        if ns is not None:
            spec.session = ns

    spec.metric = normalize_metric_alias(spec.metric, question)

    spec = apply_open_ended_date_logic(spec, question)

    iso_pat = r"^\\d{4}-\\d{2}-\\d{2}$"
    if spec.date_start != "__MISSING__" and not re.match(iso_pat, spec.date_start):
        spec.date_start = parse_date_to_iso(spec.date_start)
    if spec.date_end != "__MISSING__" and not re.match(iso_pat, spec.date_end):
        spec.date_end = parse_date_to_iso(spec.date_end)

    return spec
