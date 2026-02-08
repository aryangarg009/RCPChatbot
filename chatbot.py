# chatbot.py
import json
from typing import Any, Dict, Optional

from config import CSV_PATH
from date_io import load_data
from chat_service import process_question_with_fallback


def _print_query_spec(data: Optional[Dict[str, Any]]) -> None:
    if not data:
        return
    spec = data.get("spec")
    if spec:
        print(f"\n[QuerySpec used]: {spec}")


def _print_raw_rows(results: Optional[list]) -> None:
    if not results:
        return
    print("\nRaw rows (audit trail):")
    for r in results:
        print(r)


def _print_compare_rows(compare: Dict[str, Any]) -> None:
    if not compare:
        return
    if "session_earlier" in compare and "rows_earlier" in compare:
        print(f"\nRaw rows (earlier session: {compare['session_earlier']}):")
        for r in compare.get("rows_earlier", []):
            print(r)
    if "session_later" in compare and "rows_later" in compare:
        print(f"\nRaw rows (later session: {compare['session_later']}):")
        for r in compare.get("rows_later", []):
            print(r)


def _print_fallback_payload(result: Dict[str, Any]) -> None:
    if not result:
        return
    if "data" in result:
        print("\nData:")
        print(json.dumps(result["data"], indent=2))
    if result.get("warnings"):
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"- {w}")


def _render_response(resp: Dict[str, Any]) -> None:
    rtype = resp.get("type")
    answer = resp.get("answer", "")
    data = resp.get("data") or {}

    if rtype == "reset":
        print(f"\n[OK] {answer}\n")
        return

    if rtype == "definition":
        print(f"\n{answer}\n")
        return

    if rtype == "error":
        print("\n[BLOCKED] " + answer + "\n")
        return

    _print_query_spec(data)

    print("\nAnswer:")
    print(answer)

    if rtype == "point":
        _print_raw_rows(data.get("results"))
        print("")
        return

    if rtype == "timeseries":
        summary = data.get("summary") or {}
        if summary.get("error"):
            _print_raw_rows(data.get("results"))
            print("")
            return
        if summary.get("per_date_summary"):
            print("Per-date:")
            for row in summary.get("per_date_summary", []):
                print(row)
        _print_raw_rows(data.get("results"))
        print("")
        return

    if rtype == "session_range":
        summary = data.get("summary") or {}
        if summary.get("error"):
            _print_raw_rows(data.get("results"))
            print("")
            return
        if summary.get("per_session_summary"):
            print("Per-session:")
            for row in summary.get("per_session_summary", []):
                print(row)
        _print_raw_rows(data.get("results"))
        print("")
        return

    if rtype == "compare":
        _print_compare_rows(data.get("compare") or {})
        print("")
        return

    if rtype == "code_fallback":
        print("\n[CODE FALLBACK]")
        _print_fallback_payload((data.get("result") or {}))
        print("")
        return


def main() -> None:
    print("Loading CSV...")
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print("Ask a question. Type 'quit' to exit.\n")

    context: Optional[Dict[str, Any]] = None

    while True:
        q = input("You: ").strip()
        ql = q.lower()
        if ql in {"quit", "exit"}:
            break

        resp = process_question_with_fallback(q, df, context)
        context = resp.get("context")
        _render_response(resp)


if __name__ == "__main__":
    main()
