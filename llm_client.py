# llm_client.py
import json
import re

import httpx
from pydantic import ValidationError

from config import (
    LMSTUDIO_URL, MODEL,
    ALLOWED_METRICS, ALLOWED_GAMES, ALLOWED_SESSIONS
)
from schema import QuerySpec
from date_io import parse_date_to_iso, apply_open_ended_date_logic, extract_dates_from_text
from query_engine import normalize_session_string  # safe, no circular import
from context import normalize_metric_alias

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

def normalize_llm_obj(obj: dict) -> dict:
    """
    Convert missing/None LLM outputs into our explicit __MISSING__ placeholders
    so QuerySpec validation never crashes before follow-up context can apply.
    """
    obj.setdefault("action", "get_metric_timeseries")
    obj.setdefault("patient_id", "__MISSING__")
    obj.setdefault("metric", "__MISSING__")
    obj.setdefault("date_start", "__MISSING__")
    obj.setdefault("date_end", "__MISSING__")
    obj.setdefault("game", None)
    obj.setdefault("session", None)
    obj.setdefault("return_columns", ["date", "patient_id", "metric_value"])

    if obj.get("patient_id") is None:
        obj["patient_id"] = "__MISSING__"
    if obj.get("metric") is None:
        obj["metric"] = "__MISSING__"
    if obj.get("date_start") is None:
        obj["date_start"] = "__MISSING__"
    if obj.get("date_end") is None:
        obj["date_end"] = "__MISSING__"
    if isinstance(obj.get("session"), list):
        obj["session"] = "__MULTI__"

    return obj

def llm_question_to_query(question: str) -> QuerySpec:
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
- patient_id must be the exact string like "45_M" if mentioned.
- If session is null, date_start must be present. date_end may be "__MISSING__" for open-ended queries like "since <date>".
- If a session is specified and the question does not include dates, set date_start and date_end to "__MISSING__".
- If game/session not specified, set them to null.
- return_columns must be exactly: ["date","patient_id","metric_value"].
- If the question mentions a game like "game0", "game1", "game2", or "game3", set "game" to that exact string (case-sensitive). Otherwise set "game" to null.
- Do NOT guess the game. Only set it if explicitly mentioned in the user question.
- If the question mentions a session like "session_1", "session_2", etc., set "session" to that exact string (case-sensitive). Otherwise set "session" to null.
- Do NOT guess the session. Only set it if explicitly mentioned in the user question.
- If the question mentions MORE THAN ONE game, set "game" to "__MULTI__".
- If the question mentions MORE THAN ONE session, set "session" to "__MULTI__".

If the question is missing patient_id or metric, output:
{{"action":"get_metric_timeseries","patient_id":"__MISSING__","metric":"__MISSING__","date_start":"__MISSING__","date_end":"__MISSING__","game":null,"session":null,"return_columns":["date","patient_id","metric_value"]}}

If a session is explicitly specified in the question and dates are not mentioned,
it is allowed for date_start and date_end to be "__MISSING__".
""".strip()

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(LMSTUDIO_URL, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    json_text = extract_json_strict(content)
    obj = json.loads(json_text)
    obj = normalize_llm_obj(obj)

    # Validate schema (hard guardrail)
    spec = QuerySpec(**obj)

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

    # Validate session explicitly
    if spec.session is not None and spec.session != "__MULTI__" and spec.session not in ALLOWED_SESSIONS:
        raise ValueError(f"Session '{spec.session}' not allowed. Must be one of {ALLOWED_SESSIONS}.")

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
