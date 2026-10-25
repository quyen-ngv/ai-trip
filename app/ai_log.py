"""Report every AI provider call to Java's `ai_api_calls` log (prompt, response, usage, latency).

Each entry point binds a sink for its job (trip, assistant turn, social video step); model code
calls `report(...)` without knowing about Java. Reporting is best effort: a failed report is logged
and never fails the job. Inline images and audio (base64) are replaced by a size marker so a row
stays small.

Java answers each report with the user's AI credit status. Once it says the credit is exhausted
(the balance fell below the allowed overdraft), `credit_exhausted()` turns true for the rest of the
bound job, and the job stops at its next checkpoint instead of calling a model again.
"""
from __future__ import annotations
import logging, re, uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

Sink = Callable[[dict[str, Any]], Awaitable[Any]]
_sink: ContextVar[Sink | None] = ContextVar("ai_call_sink", default=None)
# A dict, not a bool: reports run in tasks copied from the bound context, and a value set there
# would not reach the job's own context. Mutating the shared dict does.
_credit: ContextVar[dict[str, bool] | None] = ContextVar("ai_call_credit", default=None)
CREDIT_EXHAUSTED = "CREDIT_EXHAUSTED"
_feature: ContextVar[str] = ContextVar("ai_call_feature", default="AI_TRIP")
_trace: ContextVar[str | None] = ContextVar("ai_call_trace", default=None)
_DATA_URL = re.compile(r"data:[\w.+/-]+;base64,[A-Za-z0-9+/=]{64,}")
_BASE64_RUN = re.compile(r"^[A-Za-z0-9+/=]{512,}$")


def bind(sink: Sink | None, feature: str = "AI_TRIP", trace_id: str | None = None):
    """Route this context's reports (and tasks created from it) to `sink`, labelled `feature`
    (AI_TRIP for trip planning, ASSISTANT for Pengoo's chat turns, SOCIAL_LOCATION for social
    video extraction) and `trace_id` (the Java job a social step belongs to)."""
    _feature.set(feature)
    _trace.set(trace_id)
    _credit.set({"exhausted": False})
    return _sink.set(sink)


def credit_exhausted() -> bool:
    """True once Java said this job's user ran out of AI credits; checked before each model call."""
    state = _credit.get()
    return bool(state and state["exhausted"])


class CreditExhausted(Exception):
    """Raised at a checkpoint when the user ran out of AI credits; the job ends with CREDIT_EXHAUSTED."""

    def __init__(self) -> None:
        super().__init__(CREDIT_EXHAUSTED)


def require_credit() -> None:
    """Checkpoint before a model call: stops the job when the user ran out of AI credits."""
    if credit_exhausted():
        raise CreditExhausted()


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


def anthropic_usage(messages: Any) -> dict[str, int] | None:
    """Usage summed over raw Messages answers (one dict, or a list when a turn was resumed), in the
    log's terms: input includes cache reads and writes, which are also given apart (writes split by
    cache lifetime, since Claude bills them 1.25x / 2x), and web searches as billed."""
    answers = messages if isinstance(messages, list) else [messages]
    usages = [a.get("usage") for a in answers if isinstance(a, dict) and isinstance(a.get("usage"), dict)]
    if not usages:
        return None
    total = {"input": 0, "cached": 0, "cache_write_5m": 0, "cache_write_1h": 0, "output": 0, "searches": 0}
    for usage in usages:
        read = usage.get("cache_read_input_tokens") or 0
        split = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
        write = usage.get("cache_creation_input_tokens")
        if write is None:
            write = (split.get("ephemeral_5m_input_tokens") or 0) + (split.get("ephemeral_1h_input_tokens") or 0)
        one_hour = min(split.get("ephemeral_1h_input_tokens") or 0, write)
        total["input"] += (usage.get("input_tokens") or 0) + read + write
        total["cached"] += read
        total["cache_write_5m"] += write - one_hour
        total["cache_write_1h"] += one_hour
        total["output"] += usage.get("output_tokens") or 0
        total["searches"] += ((usage.get("server_tool_use") or {}).get("web_search_requests") or 0)
    return total


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        if _BASE64_RUN.match(value):
            return f"<base64 {len(value)} chars>"
        return _DATA_URL.sub(lambda m: f"<data-url {len(m.group())} chars>", value)
    return value


async def report(*, operation: str, provider: str, model: str, ok: bool, request: Any, response: Any = None,
                 error: Any = None, input_tokens: int | None = None, cached_input_tokens: int | None = None,
                 cache_write_5m_tokens: int | None = None, cache_write_1h_tokens: int | None = None,
                 output_tokens: int | None = None, web_search_calls: int | None = None,
                 latency_ms: int | None = None) -> None:
    sink = _sink.get()
    if sink is None:
        return
    entry = {
        "feature": _feature.get(), "operation": operation, "correlationId": str(uuid.uuid4()),
        "traceId": _trace.get(), "provider": provider, "model": model, "status": "SUCCEEDED" if ok else "FAILED",
        "request": _redact(request), "response": _redact(response), "error": error,
        "inputTokens": input_tokens, "cachedInputTokens": cached_input_tokens,
        "cacheWrite5mTokens": cache_write_5m_tokens, "cacheWrite1hTokens": cache_write_1h_tokens,
        "outputTokens": output_tokens,
        "webSearchCalls": web_search_calls, "latencyMs": latency_ms,
    }
    try:
        answer = await sink(entry)
    except Exception as e:  # never let logging break the job
        logger.warning("AI call log not delivered (%s %s): %s", operation, model, e)
        return
    state = _credit.get()
    if state is not None and isinstance(answer, dict) and answer.get("creditExhausted"):
        state["exhausted"] = True
        logger.info("AI credit exhausted after %s (balance %s)", operation, answer.get("creditBalance"))


def unbind(binding) -> None:
    _sink.reset(binding)
