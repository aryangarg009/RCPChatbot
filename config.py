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
    "avg_path_ratio",
    "avg_mean_dev",
    "avg_max_dev",
    "avg_actual_len",
    "avg_ideal_len",
    "avg_excess_len",
    "f_patient",
    "avg_f_ee",
    "diff_from_prev",
    "improvement",
    "area",
])

ALLOWED_RETURN_COLUMNS = sorted([
    "date", "patient_id", "patient", "game", "session",
    *ALLOWED_METRICS
])

ALLOWED_GAMES = [f"game{i}" for i in range(0, 3)]
ALLOWED_SESSIONS = [f"session_{i}" for i in range(1, 20000)]

FOLLOWUP_CUES = [
    "what about", "how about", "and", "also", "their", "that one", "same", "instead"
]

RESET_COMMANDS = {"reset", "reset context", "clear", "clear context", "new question"}
