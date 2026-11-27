"""
The progress events a search streams to its client, as plain dicts.

The same five types the web client reads from the text search's server-sent events
(status, agent_step, chunk, done, error); the extension's sidecar sends them over stdio.
"""
from typing import Any, Dict

STATUS, AGENT_STEP, CHUNK, DONE, ERROR = "status", "agent_step", "chunk", "done", "error"


def status(stage: str, message: str) -> Dict[str, Any]:
    """A stage such as PLANNING, RETRIEVING, ANALYZING, EDITING, with a sentence for people."""
    return {"type": STATUS, "stage": stage, "message": message}


def agent_step(agent: str, detail: str) -> Dict[str, Any]:
    """One workflow node finished: what it did."""
    return {"type": AGENT_STEP, "agent": agent, "detail": detail}


def chunk(content: str) -> Dict[str, Any]:
    """A piece of the answer text, as the model writes it."""
    return {"type": CHUNK, "content": content}


def done(**result: Any) -> Dict[str, Any]:
    return {"type": DONE, **result}


def error(message: str) -> Dict[str, Any]:
    return {"type": ERROR, "message": message}
