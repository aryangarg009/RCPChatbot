# Patient Chatbot (Deterministic Backend)

This repo is a deterministic backend for patient rehab metrics.
The LLM is used only to parse a user question into a strict JSON query spec.
All data processing and answers are computed in Python.

## Quick Start

### 1) Install Python
You need Python 3.10 or higher.

### 2) One-time setup (from the repo folder)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pandas python-dateutil httpx pydantic
cp .env.example .env
```

### 3) Choose how questions are parsed
You have two options. Pick one.

#### Option A: OpenAI only (recommended if you do NOT want LM Studio)
1. Open `config.py` and set:
   ```python
   PARSER_BACKEND = "openai"
   ```
2. Set your parser key in `.env`:
   ```dotenv
   OPENAI_API_KEY_PARSER="your_parser_key_here"
   ```
   Backward-compatible fallback:
   ```dotenv
   OPENAI_API_KEY="your_key_here"
   ```
3. Done. This uses OpenAI to parse the question, then Python computes the answer deterministically.

#### Option B: LM Studio + OpenAI fallback (local parser + cloud fallback)
1. Install and run LM Studio.
2. Load a model in LM Studio (the model id must match `MODEL`).
3. Open `config.py` and set:
   ```python
   PARSER_BACKEND = "lmstudio"
   ```
4. In `config.py`, uncomment and set:
   ```python
   LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
   MODEL = "qwen2.5-7b-instruct"
   ```
5. Optional but recommended: Set your fallback key in `.env`:
   ```dotenv
   OPENAI_API_KEY_FALLBACK="your_fallback_key_here"
   ```
   Backward-compatible fallback:
   ```dotenv
   OPENAI_API_KEY="your_key_here"
   ```

### 4) Run the API (web/backend)
```bash
uvicorn api:app --reload --port 8000
```
Use this endpoint:
```
POST http://localhost:8000/chat
```
Example request:
```json
{"message":"how has patient 46 range of motion changed from 2022-11-07 to 2022-11-10 in game0?","context":null}
```

### 5) Run the CLI (local terminal)
```bash
python chatbot.py
```

### 6) Quick test (API)
```bash
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"how has patient 46 range of motion changed from 2022-11-07 to 2022-11-10 in game0?"}'
```

## Notes
- CSV source: `Combined_AllMetrics.csv`
- Parser settings: `config.py`
- This project auto-loads `.env` from the repo root for CLI and API runs.
- LLM never sees CSV data. It only outputs JSON. All numbers are computed deterministically.

## Query Run Logging (CSV)
Every query run is logged automatically (CLI and API) to:
- `query_metrics_log.csv` (repo root)

Columns:
- `input_query`
- `latency_ms`
- `path` (`deterministic` or `code_fallback`)
- `type` (for example: `timeseries`, `point`, `session_range`, `compare`, `error`, `code_fallback`)
- `output` (final answer text returned to the user)

Optional overrides in `.env`:
```dotenv
ENABLE_QUERY_LOG_CSV=true
QUERY_LOG_CSV_PATH="query_metrics_log.csv"
```

## Provider-Managed Code Fallback (OpenAI)
This repo can optionally fall back to OpenAI's code interpreter when the strict parser
can't answer a question. This is disabled only if `ENABLE_CODE_FALLBACK = False`.

### Setup
Set your fallback key in `.env`:
```dotenv
OPENAI_API_KEY_FALLBACK="your_fallback_key_here"
```
Backward-compatible fallback:
```dotenv
OPENAI_API_KEY="your_key_here"
```

### What happens
- The primary deterministic path runs first.
- If it returns an error and the fallback policy allows it, the system calls OpenAI's
  code interpreter with the CSV attached.
- The response type is `code_fallback` with a structured result in `data.result`.
