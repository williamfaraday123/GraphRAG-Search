"""Minimal client for an OpenAI-compatible /chat/completions endpoint (Ollama, vLLM, OpenAI).

Uses only the standard library. The `openai` package (pulled in by langchain_openai)
loads jiter's compiled DLL, which Windows Smart App Control blocks with
"DLL load failed while importing jiter: An Application Control policy has blocked this file".
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Optional

DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))


@dataclass
class ChatResponse:
    content: str


def _role(message) -> str:
    kind = getattr(message, "type", None) or (message.get("role") if isinstance(message, dict) else None)
    return {"human": "user", "ai": "assistant"}.get(kind, kind or "user")


def _content(message) -> str:
    return message.get("content", "") if isinstance(message, dict) else message.content


class ChatClient:
    """Drop-in for the parts of langchain's ChatOpenAI this project used:
    ChatClient(model=..., base_url=..., api_key=..., temperature=...).invoke(messages).content
    Messages can be langchain message objects or {"role", "content"} dicts."""

    def __init__(self, model: str, base_url: str, api_key: str = "ollama",
                 temperature: Optional[float] = None, timeout: float = DEFAULT_TIMEOUT):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def _request(self, messages, stream: bool) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [{"role": _role(m), "content": _content(m)} for m in messages],
            "stream": stream,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )

    def _open(self, request):
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"LLM request to {self.url} failed ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM server unreachable at {self.url}: {e.reason}") from e

    def invoke(self, messages) -> ChatResponse:
        with self._open(self._request(messages, stream=False)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return ChatResponse(content=payload["choices"][0]["message"].get("content") or "")

    def stream(self, messages) -> Iterator[str]:
        """Yields the reply's text as the model writes it (server-sent events:
        'data: {json}' lines, ending with 'data: [DONE]')."""
        with self._open(self._request(messages, stream=True)) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece


def chat_client(model_env: str, temperature: Optional[float] = None) -> ChatClient:
    return ChatClient(
        model=os.getenv(model_env, "qwen3:4b-instruct"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        temperature=temperature,
    )
