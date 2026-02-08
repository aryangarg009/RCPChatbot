# config.py
CSV_PATH = "Combined_AllMetrics.csv"
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen2.5-7b-instruct"  # must match LM Studio "API identifier"

# Provider-managed code execution fallback (OpenAI Responses API)
ENABLE_CODE_FALLBACK = True
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1"
OPENAI_CODE_CONTAINER_MEMORY = "1g"  # 1g|4g|16g|64g

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
