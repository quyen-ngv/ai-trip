"""OpenAI-compatible chat model with tool calling, for the assistant.

Separate from `llm.CachedModel` on purpose: a chat turn must never be served from a cache, and it
needs the whole message list plus tool definitions rather than one system/user pair. Provider
quirks are handled the same way as in `llm.py` (one retry per rejected parameter), so OpenAI,
DeepSeek and Anthropic's OpenAI-compatible endpoint all work from configuration alone.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .. import ai_log

logger = logging.getLogger(__name__)

PREFIXES = ("ASSISTANT_LLM", "QUALITY_LLM", "CHEAP_LLM")
TIMEOUT_SECONDS = float(os.getenv("ASSISTANT_LLM_TIMEOUT_SECONDS", "60") or 60)


def _configured(prefix: str) -> bool:
    return all(os.getenv(f"{prefix}_{key}", "") for key in ("BASE_URL", "API_KEY", "MODEL"))


class ChatModel:
    """The first fully configured of ASSISTANT_LLM_*, QUALITY_LLM_*, CHEAP_LLM_*."""

    def __init__(self, prefixes: tuple[str, ...] = PREFIXES):
        chosen = next((prefix for prefix in prefixes if _configured(prefix)), prefixes[0])
        self.prefix = chosen
        self.url = os.getenv(f"{chosen}_BASE_URL", "").rstrip("/")
        self.key = os.getenv(f"{chosen}_API_KEY", "")
        self.model = os.getenv(f"{chosen}_MODEL", "")

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key and self.model)

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 1800,
                       temperature: float = 0.4, operation: str = "CHAT", cache_key: str | None = None
                       ) -> dict | None:
        """One model round. Returns the assistant message ({content, tool_calls}) or None."""
        if not self.configured:
            logger.error("assistant model not configured (%s)", self.prefix)
            return None
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if cache_key and "openai.com" in self.url:
            body["prompt_cache_key"] = cache_key
        started = time.monotonic()
        trace: dict[str, Any] = {}
        message = await self._post(body, trace)
        data = trace.get("response")
        input_tokens, cached, output_tokens = ai_log.chat_usage(data)
        await ai_log.report(operation=operation, provider=ai_log.provider_of(self.url), model=self.model,
                            ok=message is not None, request=trace.get("request"), response=data,
                            error=trace.get("error"), input_tokens=input_tokens, cached_input_tokens=cached,
                            output_tokens=output_tokens, latency_ms=int((time.monotonic() - started) * 1000))
        if message is not None:
            message["_usage"] = {"input": input_tokens or 0, "output": output_tokens or 0}
        return message

    async def _post(self, body: dict, trace: dict) -> dict | None:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.post(f"{self.url}/chat/completions", headers=headers, json=body)
                for _ in range(4):
                    if response.status_code != 400:
                        break
                    error = response.text.lower()
                    if "max_tokens" in error and "max_tokens" in body:
                        body["max_completion_tokens"] = body.pop("max_tokens")
                    elif "temperature" in error and "temperature" in body:
                        body = {k: v for k, v in body.items() if k != "temperature"}
                    elif "prompt_cache_key" in error and "prompt_cache_key" in body:
                        body = {k: v for k, v in body.items() if k != "prompt_cache_key"}
                    else:
                        break
                    response = await client.post(f"{self.url}/chat/completions", headers=headers, json=body)
            trace["request"] = body
            if response.status_code != 200:
                trace["error"] = {"httpStatus": response.status_code, "body": response.text[:4000]}
                logger.error("assistant LLM HTTP %s: %s", response.status_code, response.text[:400])
                return None
            data = response.json()
            trace["response"] = data
            choices = data.get("choices") or []
            if not choices:
                trace["error"] = {"reason": "no choices"}
                return None
            message = choices[0].get("message") or {}
            return {"content": message.get("content") or "", "tool_calls": message.get("tool_calls") or [],
                    "finish_reason": choices[0].get("finish_reason")}
        except Exception as exc:  # network, timeout, bad JSON
            trace.setdefault("request", body)
            trace["error"] = {"reason": type(exc).__name__, "message": str(exc)}
            logger.error("assistant LLM error: %s: %s", type(exc).__name__, exc)
            return None
