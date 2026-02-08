import json
import os
import re
from typing import Any, Dict, Optional

import httpx

from config import (
    OPENAI_API_BASE,
    OPENAI_MODEL,
    OPENAI_CODE_CONTAINER_MEMORY,
)


class OpenAIFallbackError(RuntimeError):
    pass


def _extract_json_strict(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise OpenAIFallbackError("Model did not return a single JSON object.")
    return text


def _extract_output_text(resp_json: Dict[str, Any]) -> str:
    # Responses API: output -> message -> content -> output_text
    for item in resp_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    # Fallback: some SDKs surface output_text
    return resp_json.get("output_text", "")


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIFallbackError("OPENAI_API_KEY is not set.")
    return api_key


def _upload_csv(csv_path: str, api_key: str) -> str:
    if not os.path.exists(csv_path):
        raise OpenAIFallbackError(f"CSV not found at {csv_path}.")

    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=60.0) as client:
        with open(csv_path, "rb") as f:
            files = {"file": (os.path.basename(csv_path), f, "text/csv")}
            data = {"purpose": "user_data"}
            r = client.post(f"{OPENAI_API_BASE}/files", headers=headers, data=data, files=files)
            r.raise_for_status()
            return r.json()["id"]


def run_code_fallback(
    question: str,
    csv_path: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    api_key = _get_api_key()
    file_id = _upload_csv(csv_path, api_key)

    instructions = (
        "You are a data analyst. Use the python tool to answer the user's question "
        "based solely on the attached CSV. The CSV is available in the code interpreter "
        "container (typically under /mnt/data). Return ONLY one JSON object with keys: "
        "`answer` (string), `data` (object with any computed tables/values), "
        "`confidence` (0-1), and `warnings` (array of strings)."
    )

    user_prompt = {
        "question": question,
        "context": context or {},
        "csv_filename_hint": os.path.basename(csv_path),
        "notes": "Load the CSV from /mnt/data. If multiple files exist, list and choose the CSV.",
    }

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "tool_choice": "required",
        "tools": [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "memory_limit": OPENAI_CODE_CONTAINER_MEMORY,
                    "file_ids": [file_id],
                },
            }
        ],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_prompt)},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{OPENAI_API_BASE}/responses", headers=headers, json=payload)
        r.raise_for_status()
        resp_json = r.json()

    output_text = _extract_output_text(resp_json)
    if not output_text:
        raise OpenAIFallbackError("No output_text found in Responses API output.")

    json_text = _extract_json_strict(output_text)
    result = json.loads(json_text)
    if not isinstance(result, dict):
        raise OpenAIFallbackError("Fallback response was not a JSON object.")

    return result
