"""OpenAI-compatible chat model with tool calling, for the assistant.

Separate from `llm.CachedModel` on purpose: a chat turn must never be served from a cache, and it
needs the whole message list plus tool definitions rather than one system/user pair. The call itself
goes through `providers.chat`, so OpenAI, DeepSeek and Anthropic's OpenAI-compatible endpoint all
work from configuration alone.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .. import providers

logger = logging.getLogger(__name__)

PREFIXES = ("ASSISTANT_LLM", "QUALITY_LLM", "CHEAP_LLM")
TIMEOUT_SECONDS = float(os.getenv("ASSISTANT_LLM_TIMEOUT_SECONDS", "60") or 60)
# A chat answer is short and tool-driven: at the default effort a third of Pengoo's output tokens
# were thinking (prod log, 2026-10-25). "low" also makes the model consolidate its tool calls.
# OpenAI endpoints that reject the field with tools are handled by providers._adapt. Empty = unset.
REASONING_EFFORT = os.getenv("ASSISTANT_LLM_REASONING_EFFORT", "low").strip().lower()
# Claude keeps the static prefix (rules + tools, ~10k tokens) this long. Questions in one chat come
# 6-60 minutes apart more often than within 5, so the 2x write pays back from the third question.
CACHE_TTL = os.getenv("ASSISTANT_CACHE_TTL", "1h").strip().lower()


class ChatModel:
    """The first fully configured of ASSISTANT_LLM_*, QUALITY_LLM_*, CHEAP_LLM_*."""

    def __init__(self, prefixes: tuple[str, ...] = PREFIXES):
        self.endpoint = providers.from_tiers(*prefixes)
        self.prefix = self.endpoint.name

    @property
    def configured(self) -> bool:
        return self.endpoint.configured

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 1800,
                       temperature: float = 0.4, operation: str = "CHAT", cache_key: str | None = None
                       ) -> dict | None:
        """One model round. Returns the assistant message ({content, tool_calls}) or None."""
        if not self.configured:
            logger.error("assistant model not configured (%s)", self.prefix)
            return None
        body: dict[str, Any] = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if cache_key and "openai.com" in self.endpoint.base_url:
            body["prompt_cache_key"] = cache_key
        if REASONING_EFFORT:
            body["reasoning_effort"] = REASONING_EFFORT
        if CACHE_TTL in ("1h", "5m") and self.endpoint.is_claude:
            body["cache_ttl"] = CACHE_TTL
        result = await providers.chat(self.endpoint, body, operation=operation, timeout=TIMEOUT_SECONDS)
        if not result.ok:
            return None
        usage = result.usage
        return {**result.value, "_usage": {"input": usage["inputTokens"] or 0, "output": usage["outputTokens"] or 0}}
