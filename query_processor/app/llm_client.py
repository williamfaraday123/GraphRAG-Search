"""The chat client for the models in .env (EDGE_MODEL, ...). The client itself is shared with the
VS Code extension (core/graphrag_core/llm.py): an OpenAI-compatible /chat/completions client
that uses only the standard library."""
import os
from typing import Optional

from graphrag_core.llm import ChatClient, ChatResponse  # noqa: F401 (re-exported)


def chat_client(model_env: str, temperature: Optional[float] = None) -> ChatClient:
    return ChatClient(
        model=os.getenv(model_env, "qwen3:4b-instruct"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        temperature=temperature,
    )
