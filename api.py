# api.py
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_service import process_question
from config import CSV_PATH
from date_io import load_data


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


app = FastAPI()

# Dev-friendly CORS; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data once on startup.
_DF = load_data(CSV_PATH)


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    return process_question(req.message, _DF, req.context)
