# date_io.py
import re
from datetime import date
from typing import List

import pandas as pd
from dateutil import parser as dateparser

from schema import QuerySpec

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

def today_iso() -> str:
    return date.today().isoformat()

def parse_date_to_iso(d: str) -> str:
    """
    Accepts dates like 10/3/22, 10-03-2022, 2022-03-10,
    or ISO datetime strings like 2022-11-07T09:51:02.000.
    Returns ISO date YYYY-MM-DD.
    """
    s = d.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
        dt = dateparser.isoparse(s)
    else:
        dt = dateparser.parse(s, dayfirst=True)
    return dt.date().isoformat()

def extract_dates_from_text(question: str) -> List[str]:
    """
    Extract date strings like 1/3/22, 2022-03-01, 2022-11-07T09:51:02.000,
    or "7th November 2022" from the question.
    Returns them in the order they appear (best effort).
    """
    q = question.strip()
    found = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(q):
            found.append((m.start(), m.group(0)))
    return [s for _, s in sorted(found, key=lambda x: x[0])]

def apply_open_ended_date_logic(spec: QuerySpec, question: str) -> QuerySpec:
    """
    If user gives ONE date and implies open-ended range ("since", "from", etc.),
    set date_end to today. Also deterministically parses D/M/Y using parse_date_to_iso.
    If user provides TWO dates, deterministically set start/end based on text order.
    """
    ql = question.lower()
    dates = extract_dates_from_text(question)

    open_ended_cues = ["since", "from", "starting", "start from", "after"]
    has_open_ended_cue = any(cue in ql for cue in open_ended_cues)

    if len(dates) == 1 and has_open_ended_cue:
        spec.date_start = parse_date_to_iso(dates[0])  # D/M/Y enforced
        spec.date_end = today_iso()
        return spec

    if len(dates) >= 2:
        spec.date_start = parse_date_to_iso(dates[0])
        spec.date_end = parse_date_to_iso(dates[1])
        return spec

    return spec

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize key string columns to avoid hidden whitespace mismatches
    for col in ["patient", "game", "session"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")

    # Keep raw date for debugging
    df["_date_raw"] = df["date"].astype(str).str.strip()

    # ---- parse ISO datetimes separately to avoid day/month flipping ----
    raw = df["_date_raw"]
    iso_mask = raw.str.match(r"^\d{4}-\d{2}-\d{2}T")
    iso_date_mask = raw.str.match(r"^\d{4}-\d{2}-\d{2}$")

    dt_iso = pd.to_datetime(raw.where(iso_mask | iso_date_mask), errors="coerce")
    dt_other = pd.to_datetime(raw.where(~(iso_mask | iso_date_mask)), errors="coerce", dayfirst=True)

    dt = dt_iso.fillna(dt_other)
    if hasattr(dt, "dt"):
        dt = dt.dt.normalize()
    df["date"] = dt

    bad = df["date"].isna().sum()
    if bad > 0:
        print(f"[WARN] {bad} rows have unparseable dates and will be ignored in date filtering.")

    return df
