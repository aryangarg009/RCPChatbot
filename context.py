# context.py
import re
from typing import Optional

from config import FOLLOWUP_CUES, ALLOWED_METRICS
from schema import QuerySpec

_ALIAS_SPLIT_RE = re.compile(r"[_-]+")
_ALIAS_WS_RE = re.compile(r"\s+")

# Raw alias phrases → canonical metric names
_RAW_METRIC_ALIAS_MAP = {
    "sparc": "average_sparc",
    "smoothness": "average_sparc",

    "efficiency": "avg_efficiency",
    "efficient": "avg_efficiency",

    "path ratio": "avg_path_ratio",
    "pathratio": "avg_path_ratio",

    "mean deviation": "avg_mean_dev",
    "max deviation": "avg_max_dev",

    "force": "f_patient",
    "strength": "f_patient",
    "f_patient": "f_patient",
    "avg_f_patient": "f_patient",

    "area": "area",
    "range of motion": "area",
    "rangeofmotion": "area",
    "rom": "area",
}

# Normalize alias keys once for consistent matching
def _normalize_alias_text(text: str) -> str:
    lowered = text.lower()
    spaced = _ALIAS_SPLIT_RE.sub(" ", lowered)
    return _ALIAS_WS_RE.sub(" ", spaced).strip()

METRIC_ALIAS_MAP = {_normalize_alias_text(k): v for k, v in _RAW_METRIC_ALIAS_MAP.items()}

def looks_like_followup(question: str) -> bool:
    q = question.strip().lower()
    return any(cue in q for cue in FOLLOWUP_CUES)

def question_mentions_patient(question: str) -> bool:
    return re.search(r"\b\d+_[MF]\b", question) is not None

def question_mentions_game(question: str) -> bool:
    return re.search(r"\bgame[0-9]+\b", question) is not None

def question_mentions_session(question: str) -> bool:
    return re.search(r"\bsession[_\s]*\d+\b", question.lower()) is not None

def question_mentions_dates(question: str) -> bool:
    return (
        re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", question) is not None
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", question) is not None
    )

def extract_metric_from_text(question: str) -> Optional[str]:
    """
    Deterministically detect a metric name or alias mentioned in the user's text.
    Safe because we only return values from ALLOWED_METRICS via allowlisted mapping.
    """
    q = question.lower()
    q_norm = _normalize_alias_text(question)

    # 1) Exact metric tokens (e.g. "avg_efficiency")
    for m in ALLOWED_METRICS:
        if re.search(rf"\b{re.escape(m.lower())}\b", q):
            return m

    # 2) Alias phrases → canonical metric names (SAFE allowlist mapping)
    for phrase, metric in METRIC_ALIAS_MAP.items():
        if " " in phrase:
            if phrase in q_norm:
                return metric
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", q_norm):
                return metric

    return None

def normalize_metric_alias(metric: str, question: Optional[str] = None) -> str:
    """
    Map metric aliases (e.g., "rom", "range_of_motion") to canonical metric names.
    If question is provided, prefer the explicit alias found in the question.
    """
    if metric in ALLOWED_METRICS or metric == "__MISSING__":
        return metric

    if question:
        explicit = extract_metric_from_text(question)
        if explicit is not None:
            return explicit

    norm_metric = _normalize_alias_text(metric)
    mapped = METRIC_ALIAS_MAP.get(norm_metric)
    return mapped if mapped is not None else metric


def is_metric_definition_question(question: str) -> bool:
    q = question.lower().strip()
    return (
        q.startswith("what is ")
        or q.startswith("what's ")
        or "what does" in q
        or "mean?" in q
        or "meaning of" in q
        or "define" in q
        or "explain" in q
    )

def extract_metric_or_alias_from_definition_question(question: str) -> Optional[str]:
    """
    Reuse your existing deterministic extractor.
    If user says "what is sparc", extract_metric_from_text() will return average_sparc.
    """
    return extract_metric_from_text(question)

def apply_followup_context(
    new_spec: QuerySpec,
    question: str,
    last_spec: Optional[QuerySpec]
) -> QuerySpec:
    """
    Fill missing fields from last_spec ONLY when:
    - last_spec exists, AND
    - the user did not explicitly mention a new value, AND
    - new_spec has __MISSING__ (or None for game/session)
    """
    if last_spec is None:
        return new_spec

    # Metric
    explicit_metric = extract_metric_from_text(question)
    if explicit_metric is not None:
        new_spec.metric = explicit_metric
    elif new_spec.metric == "__MISSING__":
        new_spec.metric = last_spec.metric

    # Patient
    if new_spec.patient_id == "__MISSING__" and not question_mentions_patient(question):
        new_spec.patient_id = last_spec.patient_id

    # Dates
    if (new_spec.date_start == "__MISSING__" and new_spec.date_end == "__MISSING__") and not question_mentions_dates(question):
        new_spec.date_start = last_spec.date_start
        new_spec.date_end = last_spec.date_end

    # Game
    if new_spec.game is None and not question_mentions_game(question):
        new_spec.game = last_spec.game

    # Session
    if question_mentions_dates(question) and not question_mentions_session(question):
        # If the user gave dates, do not carry over a prior session.
        new_spec.session = None
    elif new_spec.session is None and not question_mentions_session(question):
        new_spec.session = last_spec.session

    return new_spec
