# Patient Chatbot (Deterministic Backend)

This repo is a deterministic backend for patient rehab metrics.
The LLM is used only to parse a user question into a strict JSON query spec.
All data processing and answers are computed in Python.

## Quick Checklist

### Prereqs
- LM Studio running locally (OpenAI-compatible server on `http://127.0.0.1:1234`)
- Model loaded in LM Studio (see `config.py` for `MODEL`)
- Python 3.10+ recommended

### One-time setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn
```

### Run the HTTP API (recommended for frontend)
```bash
uvicorn api:app --reload --port 8000
```
API endpoint:
```
POST http://localhost:8000/chat
```
Request:
```json
{"message":"your question","context":{...}}
```
Response:
```json
{"type":"timeseries|point|compare|session_range|error|reset","answer":"...","data":{...},"context":{...}}
```

### Run the CLI (local only)
```bash
python chatbot.py
```

### Quick test (API)
```bash
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"how has patient 45 m range of motion changed from 10/3/22 to 25/3/22 in game0?"}'
```

## Notes
- CSV source: `Combined_AllMetrics.csv`
- LLM endpoint + model: `config.py`
- LLM never sees CSV data. It only outputs JSON. All numbers are computed deterministically.

