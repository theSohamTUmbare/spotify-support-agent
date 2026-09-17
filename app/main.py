"""FastAPI backend for the interactive demo.

Exposes the agent's full, auditable decision for one message. The frontend renders every
stage — intent, evidence, draft, verification, and the gate-by-gate autonomy decision —
so a reviewer can see not just *what* the agent replied but *why it was (or wasn't)
allowed to send it*.

Run:  uvicorn app.main:app --reload  (from the repo root)
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import build_default_agent  # noqa: E402

app = FastAPI(title="Spotify Support Agent", version="1.0")
_agent = None


def agent():
    global _agent
    if _agent is None:
        _agent = build_default_agent(cache=True)
    return _agent


class Query(BaseModel):
    message: str


@app.post("/api/handle")
def handle(q: Query):
    try:
        res = agent().handle(q.message)
        return res.to_dict()
    except Exception as e:  # LLM unavailable / rate-limited: fail safe, don't 500
        # The safe default when the agent cannot think is to escalate to a human.
        return {
            "error": "llm_unavailable",
            "message": q.message,
            "detail": str(e)[:200],
            "decision": {
                "action": "escalate",
                "reason": "The assistant is temporarily unavailable (model rate limit); "
                          "routing to a human. This is the fail-safe default.",
                "gates": {},
            },
        }


@app.get("/api/health")
def health():
    return {"status": "ok"}


STATIC = ROOT / "app" / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
