from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env.web")

from generation import PromptConfig
from providers import default_model_for
from routing import run_rag


STATIC_DIR = Path(__file__).resolve().parent / "web_static"
WEB_PROVIDER = "nvidia"

app = FastAPI(title="Anime RAG Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list)
    style: str = "concise"
    max_docs: int = 5


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> Dict[str, object]:
    return {
        "default_model": default_model_for(WEB_PROVIDER),
        "styles": ["concise", "detailed"],
        "max_docs": 5,
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict[str, object]:
    style = request.style.strip().lower()
    if style not in {"concise", "detailed"}:
        raise HTTPException(status_code=400, detail=f"Unsupported style: {request.style}")

    max_docs = max(1, min(request.max_docs, 8))
    selected_model = default_model_for(WEB_PROVIDER)
    config = PromptConfig(
        style=style,
        max_recs=5,
        max_docs=max_docs,
        max_history_turns=6,
    )

    history = [_model_to_dict(message) for message in request.history]
    response, docs = run_rag(
        prompt=request.prompt,
        history=history,
        config=config,
        provider=WEB_PROVIDER,
        model=selected_model,
        max_docs=max_docs,
    )
    return {
        "assistant_message": response,
        "docs": docs,
        "model": selected_model,
        "style": style,
    }


def _model_to_dict(message: ChatMessage) -> Dict[str, str]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()
