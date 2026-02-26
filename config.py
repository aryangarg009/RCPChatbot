# config.py
import os
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _load_project_env() -> None:
    """
    Load key/value pairs from project-local .env into os.environ.
    Existing environment variables are preserved.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            # Allow unquoted values followed by a comment.
            value = value.split(" #", 1)[0].strip()

        os.environ.setdefault(key, value)


_load_project_env()

CSV_PATH = "Combined_AllMetrics.csv"

# Parser backend: "openai" (default) or "lmstudio"
PARSER_BACKEND = "openai"

# LM Studio parser (uncomment and set PARSER_BACKEND="lmstudio" to use)
# LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
# MODEL = "qwen2.5-7b-instruct"  # must match LM Studio "API identifier"

# Provider-managed code execution fallback (OpenAI Responses API)
ENABLE_CODE_FALLBACK = True
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1"
OPENAI_CODE_CONTAINER_MEMORY = "1g"  # 1g|4g|16g|64g

# Query logging
ENABLE_QUERY_LOG_CSV = _env_flag("ENABLE_QUERY_LOG_CSV", True)
QUERY_LOG_CSV_PATH = os.environ.get("QUERY_LOG_CSV_PATH", "query_metrics_log.csv")

ALLOWED_METRICS = sorted([
    "average_sparc",
    "avg_efficiency",
    "avg_f_patient",
    "area",
    "timestampms",
])

ALLOWED_RETURN_COLUMNS = sorted([
    "date", "patient", "game", "session", "gender",
    *ALLOWED_METRICS,
])

ALLOWED_GAMES = [f"game{i}" for i in range(0, 11)]
ALLOWED_SESSIONS = [f"session_{i}" for i in range(1, 20000)]

FOLLOWUP_CUES = [
    "what about", "how about", "and", "also", "their", "that one", "same", "instead"
]

RESET_COMMANDS = {"reset", "reset context", "clear", "clear context", "new question"}
