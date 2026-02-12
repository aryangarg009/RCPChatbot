# context.py
import re
from typing import Optional

from config import FOLLOWUP_CUES, ALLOWED_METRICS
from schema import QuerySpec

_ALIAS_SPLIT_RE = re.compile(r"[_-]+")
_ALIAS_WS_RE = re.compile(r"\s+")

_MONTH_NAMES = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAMES}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTH_NAMES}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}\b", re.IGNORECASE),
]

_PATIENT_EXPLICIT_RE = re.compile(r"\bpatient\s*(\d+)\b", re.IGNORECASE)
_PT_EXPLICIT_RE = re.compile(r"\bpt\s*(\d+)\b", re.IGNORECASE)

_DURATION_CUES = [
    "how long",
    "duration",
    "session duration",
    "session length",
    "length of session",
]

_GENDER_CUES = [
    "gender",
    "male",
    "female",
    "sex",
]

# Raw alias phrases → canonical metric names
_RAW_METRIC_ALIAS_MAP = {
    "sparc": "average_sparc",
    "smoothness": "average_sparc",

    "efficiency": "avg_efficiency",
    "efficient": "avg_efficiency",

    "force": "avg_f_patient",
    "strength": "avg_f_patient",
    "avg_f_patient": "avg_f_patient",

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
    return extract_patient_from_text(question) is not None

def question_mentions_game(question: str) -> bool:
    return re.search(r"\bgame\s*\d+\b", question, re.IGNORECASE) is not None

def question_mentions_session(question: str) -> bool:
    return re.search(r"\bsessions?[_\s]*\d+\b", question.lower()) is not None

def question_mentions_dates(question: str) -> bool:
    return any(p.search(question) for p in _DATE_PATTERNS)

def extract_patient_from_text(question: str) -> Optional[str]:
    q = question.strip().lower()

    explicit = _PATIENT_EXPLICIT_RE.search(q) or _PT_EXPLICIT_RE.search(q)
    if explicit:
        return explicit.group(1)

    # Exclude numbers that are part of dates
    date_spans = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(q):
            date_spans.append((m.start(), m.end()))

    def _in_date_span(idx: int) -> bool:
        return any(start <= idx < end for start, end in date_spans)

    game_nums = set(re.findall(r"\bgame\s*(\d+)\b", q))
    session_nums = set(re.findall(r"\bsessions?[_\s]*(\d+)\b", q))

    candidates = []
    for m in re.finditer(r"\b\d+\b", q):
        if _in_date_span(m.start()):
            continue
        num = m.group(0)
        if num in game_nums or num in session_nums:
            continue
        candidates.append(num)

    if len(candidates) == 1:
        return candidates[0]
    return None

def is_duration_question(question: str) -> bool:
    q = question.lower()
    if "session" not in q:
        return False
    return any(cue in q for cue in _DURATION_CUES)

def is_gender_question(question: str) -> bool:
    q = question.lower()
    return any(cue in q for cue in _GENDER_CUES)

def extract_metric_from_text(question: str) -> Optional[str]:
    """
    Deterministically detect a metric name or alias mentioned in the user's text.
    Safe because we only return values from ALLOWED_METRICS via allowlisted mapping.
    """
    q = question.lower()
    q_norm = _normalize_alias_text(question)

    # 0) Special case: session duration
    if is_duration_question(question):
        return "timestampms"

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

def extract_metrics_from_text(question: str) -> list[str]:
    """
    Deterministically detect all metric names or aliases mentioned in user text.
    Returns unique canonical metric names in mention-scan order.
    """
    q = question.lower()
    q_norm = _normalize_alias_text(question)
    found: list[str] = []

    def _push(metric: str) -> None:
        if metric not in found:
            found.append(metric)

    if is_duration_question(question):
        _push("timestampms")

    for m in ALLOWED_METRICS:
        if re.search(rf"\b{re.escape(m.lower())}\b", q):
            _push(m)

    for phrase, metric in METRIC_ALIAS_MAP.items():
        if " " in phrase:
            if phrase in q_norm:
                _push(metric)
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", q_norm):
                _push(metric)

    return found

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

    followup = looks_like_followup(question)

    # Metric
    explicit_metric = extract_metric_from_text(question)
    if explicit_metric is not None:
        new_spec.metric = explicit_metric
    elif followup or new_spec.metric == "__MISSING__":
        new_spec.metric = last_spec.metric

    # Patient
    if (followup and not question_mentions_patient(question)) or (
        new_spec.patient == "__MISSING__" and not question_mentions_patient(question)
    ):
        new_spec.patient = last_spec.patient

    # Dates
    if (new_spec.date_start == "__MISSING__" and new_spec.date_end == "__MISSING__") and not question_mentions_dates(question):
        new_spec.date_start = last_spec.date_start
        new_spec.date_end = last_spec.date_end

    # Game
    if (followup and not question_mentions_game(question)) or (
        new_spec.game is None and not question_mentions_game(question)
    ):
        new_spec.game = last_spec.game

    # Session
    if question_mentions_dates(question) and not question_mentions_session(question):
        # If the user gave dates, do not carry over a prior session.
        new_spec.session = None
    elif (followup and not question_mentions_session(question)) or (
        new_spec.session is None and not question_mentions_session(question)
    ):
        new_spec.session = last_spec.session

    return new_spec
