"""Report every AI provider call to Java's `ai_api_calls` log (prompt, response, usage, latency).

`run_job` binds a sink for the job; model code calls `report(...)` without knowing about Java.
Reporting is best effort: a failed report is logged and never fails the trip.
"""
from __future__ import annotations
import logging, uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

Sink = Callable[[dict[str, Any]], Awaitable[None]]
_sink: ContextVar[Sink | None] = ContextVar("ai_call_sink", default=None)


def bind(sink: Sink | None):
    """Route this context's reports (and tasks created from it) to `sink`."""
    return _sink.set(sink)


def provider_of(base_url: str) -> str:
    host = (urlparse(base_url).hostname or base_url or "").lower()
    for name in ("deepseek", "openai", "anthropic", "google", "openrouter", "groq", "mistral"):
        if name in host:
            return name
    return host[:64]


def chat_usage(data: Any) -> tuple[int | None, int | None, int | None]:
    """(input, cached input, output) from a chat/completions body (OpenAI and DeepSeek shapes)."""
    usage = (data or {}).get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None, None, None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = usage.get("prompt_cache_hit_tokens")
    return usage.get("prompt_tokens"), cached, usage.get("completion_tokens")


def responses_usage(data: Any) -> tuple[int | None, int | None, int | None, int]:
    """(input, cached input, output, web search calls) from a Responses API body."""
    if not isinstance(data, dict):
        return None, None, None, 0
    usage = data.get("usage") or {}
    searches = sum(1 for item in data.get("output") or []
                   if isinstance(item, dict) and item.get("type") == "web_search_call")
    return (usage.get("input_tokens"), (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            usage.get("output_tokens"), searches)


async def report(*, operation: str, provider: str, model: str, ok: bool, request: Any, response: Any = None,
                 error: Any = None, input_tokens: int | None = None, cached_input_tokens: int | None = None,
                 output_tokens: int | None = None, web_search_calls: int | None = None,
                 latency_ms: int | None = None) -> None:
    sink = _sink.get()
    if sink is None:
        return
    entry = {
        "feature": "AI_TRIP", "operation": operation, "correlationId": str(uuid.uuid4()),
        "provider": provider, "model": model, "status": "SUCCEEDED" if ok else "FAILED",
        "request": request, "response": response, "error": error,
        "inputTokens": input_tokens, "cachedInputTokens": cached_input_tokens, "outputTokens": output_tokens,
        "webSearchCalls": web_search_calls, "latencyMs": latency_ms,
    }
    try:
        await sink(entry)
    except Exception as e:  # never let logging break the job
        logger.warning("AI call log not delivered (%s %s): %s", operation, model, e)


def unbind(binding) -> None:
    _sink.reset(binding)
