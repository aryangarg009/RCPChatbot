# date_io.py
import re
from datetime import date
from typing import List

import pandas as pd
from dateutil import parser as dateparser

from schema import QuerySpec

def today_iso() -> str:
    return date.today().isoformat()

def parse_date_to_iso(d: str) -> str:
    """
    Accepts dates like 10/3/22, 10-03-2022, 2022-03-10, etc.
    Returns ISO date YYYY-MM-DD.
    """
    dt = dateparser.parse(d, dayfirst=True)  # important for SG style dates
    return dt.date().isoformat()

def extract_dates_from_text(question: str) -> List[str]:
    """
    Extract date strings like 1/3/22, 01-03-2022, 2022-03-01 from the question.
    Returns them in the order they appear (best effort).
    """
    q = question.strip()

    dmy = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", q)
    iso = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)

    found = []
    for s in dmy + iso:
        idx = q.find(s)
        if idx != -1:
            found.append((idx, s))
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

    if "avg_f_patient" in df.columns and "f_patient" not in df.columns:
        df = df.rename(columns={"avg_f_patient": "f_patient"})

    # Normalize key string columns to avoid hidden whitespace mismatches
    for col in ["patient_id", "game", "session"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")

    # Keep raw date for debugging
    df["_date_raw"] = df["date"].astype(str).str.strip()

    # ---- explicit multi-format date parsing ----
    dt_iso = pd.to_datetime(df["_date_raw"], format="%Y-%m-%d", errors="coerce")
    dt_dmy2 = pd.to_datetime(df["_date_raw"], format="%d/%m/%y", errors="coerce")
    dt_dmy4 = pd.to_datetime(df["_date_raw"], format="%d/%m/%Y", errors="coerce")

    df["date"] = dt_iso.fillna(dt_dmy2).fillna(dt_dmy4)

    bad = df["date"].isna().sum()
    if bad > 0:
        print(f"[WARN] {bad} rows have unparseable dates and will be ignored in date filtering.")

    return df
