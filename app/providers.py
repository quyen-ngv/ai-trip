"""The one place in the product that talks to model providers.

Every model call — trip planning, Pengoo, social video extraction — is a POST to an
OpenAI-compatible `/chat/completions` (`chat`) or OpenAI's `/responses` (`responses`), made here and
reported here to Java's `ai_api_calls` log. Claude endpoints go through the official Anthropic SDK
(`anthropic_adapter.py`), which translates to and from these OpenAI shapes, so callers never branch
on the provider. Speech-to-text lives in `transcription.py`.

Configuration is one switch, AI_PROVIDER (claude | openai | deepseek), and per provider a key, a
main model and a cheap model:

    CLAUDE_API_KEY    CLAUDE_MODEL    CLAUDE_CHEAP_MODEL      both default to claude-haiku-5-5
    OPENAI_API_KEY    OPENAI_MODEL    OPENAI_CHEAP_MODEL      gpt-6-luna; cheap defaults to main
    DEEPSEEK_API_KEY  DEEPSEEK_MODEL  DEEPSEEK_CHEAP_MODEL    deepseek-v4-flash; cheap to main

The main model plans itineraries, answers in Pengoo and extracts places from social videos; the
cheap one (tier CHEAP_LLM) writes research queries, searches and curates the web and runs the
social topic pre-filter. A tier (CHEAP_LLM, QUALITY_LLM, ASSISTANT_LLM) can still be pinned on its
own with <PREFIX>_PROVIDER / _BASE_URL / _API_KEY / _MODEL: each value set there wins over the
switch. A social job may name its own provider and model (admin config SOCIAL_LOCATION.AI_PROVIDER /
AI_MODEL, empty by default). An endpoint on api.anthropic.com is a Claude endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from . import ai_log, anthropic_adapter

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
# Reasoning models (gpt-5.x) reject function tools on /chat/completions unless reasoning is off.
# Models seen rejecting it are remembered so later calls send "none" up front.
_NO_REASONING_MODELS: set[str] = set()


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str
    name: str = ""  # tier prefix or provider name, for logs

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @property
    def is_claude(self) -> bool:
        return "anthropic.com" in self.base_url


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


_CHEAP_TIERS = ("CHEAP_LLM",)


def _provider_name(value: str | None) -> str:
    name = (value or "").strip().upper()
    return "ANTHROPIC" if name == "CLAUDE" else name


def provider_defaults(provider: str | None = None, *, cheap: bool = False) -> Endpoint:
    """The switch's endpoint for a provider (AI_PROVIDER when none is named): its host, key and
    main or cheap model. Fields stay blank for an unknown provider or a missing key."""
    name = _provider_name(provider or _env("AI_PROVIDER"))
    if name == "ANTHROPIC":
        model = _env("CLAUDE_CHEAP_MODEL" if cheap else "CLAUDE_MODEL") or anthropic_adapter.DEFAULT_MODEL
        return Endpoint(anthropic_adapter.DEFAULT_BASE_URL, _env("CLAUDE_API_KEY") or _env("ANTHROPIC_API_KEY"),
                        model, name)
    if name == "OPENAI":
        main = _env("OPENAI_MODEL") or _env("OPENAI_CHAT_MODEL") or "gpt-6-luna"
        return Endpoint("https://api.openai.com/v1", _env("OPENAI_API_KEY"),
                        (_env("OPENAI_CHEAP_MODEL") or main) if cheap else main, name)
    if name == "DEEPSEEK":
        main = _env("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        return Endpoint((_env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"), _env("DEEPSEEK_API_KEY"),
                        (_env("DEEPSEEK_CHEAP_MODEL") or main) if cheap else main, name)
    return Endpoint("", "", "", name)


def _tier(prefix: str) -> Endpoint:
    """The switch's endpoint for this tier, with any <PREFIX>_* value set on its own taking over."""
    base = provider_defaults(_env(f"{prefix}_PROVIDER") or None, cheap=prefix in _CHEAP_TIERS)
    return Endpoint((_env(f"{prefix}_BASE_URL") or base.base_url).rstrip("/"),
                    _env(f"{prefix}_API_KEY") or base.api_key, _env(f"{prefix}_MODEL") or base.model, prefix)


def tier_configured(prefix: str) -> bool:
    return _tier(prefix).configured


def from_tiers(*prefixes: str) -> Endpoint:
    """The first fully configured tier, else the first one (unconfigured)."""
    return _tier(next((prefix for prefix in prefixes if tier_configured(prefix)), prefixes[0]))


def from_provider(provider: str | None, model: str | None = None, base_url: str | None = None, *,
                  cheap: bool = False) -> Endpoint:
    """Provider name (ANTHROPIC|CLAUDE, DEEPSEEK, OPENAI, OPENAI_COMPATIBLE; AI_PROVIDER when none,
    DEEPSEEK when that is empty too) plus optional model and host overrides; otherwise the
    provider's main model, or its cheap one with `cheap`. OPENAI_COMPATIBLE reads AI_BASE_URL,
    AI_API_KEY and AI_MODEL.

    Raises ValueError for an unknown provider and RuntimeError when a key, model or URL is missing."""
    name = _provider_name(provider or _env("AI_PROVIDER") or "DEEPSEEK")
    model = (model or "").strip()
    base_url = (base_url or "").strip()
    if name in ("ANTHROPIC", "DEEPSEEK", "OPENAI"):
        defaults = provider_defaults(name, cheap=cheap)
        model = model or defaults.model
        base_url = base_url or defaults.base_url
        api_key = defaults.api_key
    elif name == "OPENAI_COMPATIBLE":
        model = model or _env("AI_MODEL")
        base_url = base_url or _env("AI_BASE_URL")
        api_key = _env("AI_API_KEY")
    else:
        raise ValueError(f"Unsupported AI provider: {name}")
    if not api_key:
        raise RuntimeError(f"API key is required for AI provider {name}")
    if not model:
        raise RuntimeError(f"AI model is required for provider {name}")
    if not base_url:
        raise RuntimeError(f"AI base URL is required for provider {name}")
    return Endpoint(base_url.rstrip("/"), api_key, model, name)


class Rejected(Exception):
    """Raised by a `parse` callback: the provider answered 200 but the answer is unusable."""

    def __init__(self, error: dict[str, Any], retryable: bool):
        super().__init__(error.get("message") or error.get("reason") or "rejected")
        self.error, self.retryable = error, retryable


@dataclass
class Result:
    value: Any = None                  # parse(data) when the call succeeded
    data: dict | None = None           # the provider's body
    error: dict | None = None
    retryable: bool = False            # whether an identical call could succeed

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def usage(self) -> dict[str, int | None]:
        """{inputTokens, outputTokens, totalTokens} of a chat/completions body."""
        usage = (self.data or {}).get("usage") or {}
        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if total is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total = input_tokens + output_tokens
        return {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": total}


def _adapt(body: dict, error: str) -> dict | None:
    """The body without what the provider rejected, or None when the 400 is about something else.

    Newer OpenAI models want `max_completion_tokens` and reject non-default `temperature`; some
    OpenAI-compatible hosts reject `response_format`, `prompt_cache_key` or `reasoning_effort`."""
    if "reasoning_effort" in error:
        if body.get("tools") and body.get("reasoning_effort") != "none":
            _NO_REASONING_MODELS.add(body["model"])
            return {**body, "reasoning_effort": "none"}
        if "reasoning_effort" in body:
            return {k: v for k, v in body.items() if k != "reasoning_effort"}
    if "max_tokens" in error and "max_tokens" in body:
        adapted = dict(body)
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
        return adapted
    for param in ("temperature", "response_format", "prompt_cache_key"):
        if param in error and param in body:
            return {k: v for k, v in body.items() if k != param}
    return None


def message(data: dict) -> dict:
    """{content, tool_calls, finish_reason} of the first choice; Rejected when there is none."""
    choices = data.get("choices") or []
    if not choices:
        raise Rejected({"reason": "no choices"}, True)
    first = choices[0]
    msg = first.get("message") or {}
    reply = {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
             "finish_reason": first.get("finish_reason")}
    if msg.get("_anthropic_content"):
        reply["_anthropic_content"] = msg["_anthropic_content"]  # sent back verbatim (thinking blocks)
    return reply


def json_answer(data: dict) -> Any:
    """The JSON object or array the model wrote. Rejected (with finish reason and token counts) when
    the answer is cut at the token limit, empty or not JSON."""
    msg = message(data)
    content, finish = msg["content"], msg["finish_reason"]
    if finish == "length" or not content.strip():
        usage = data.get("usage") or {}
        detail = (f"finish_reason={finish} completion_tokens={usage.get('completion_tokens')} "
                  f"reasoning_tokens={(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}")
        # A cut answer would be cut at the same place again: retrying only doubles the bill.
        reason = "output truncated at the token limit" if finish == "length" else "empty content"
        raise Rejected({"reason": reason, "message": detail}, finish != "length")
    try:
        return parse_json(content)
    except json.JSONDecodeError as exc:
        raise Rejected({"reason": "answer is not JSON", "message": str(exc)}, True) from exc


def parse_json(text: str) -> Any:
    """Parse a model's JSON answer: fences stripped, then the outermost object, then a repair of
    commas models drop between lines."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = re.sub(r'([}\]"0-9])\s*\n\s*("[-A-Za-z0-9_]+":)', r"\1,\n\2", cleaned)
        repaired = re.sub(r'\b(true|false|null)\s*\n\s*("[-A-Za-z0-9_]+":)', r"\1,\n\2", repaired)
        repaired = re.sub(r"}\s*\n\s*{", "},\n{", repaired)
        if repaired == cleaned:
            raise
        logger.warning("model JSON needed repair: %s", exc)
        return json.loads(repaired)


async def chat(endpoint: Endpoint, body: dict, *, operation: str, timeout: float,
               parse: Callable[[dict], Any] = message) -> Result:
    """One /chat/completions call: adapted to parameters the model rejects, parsed, and reported."""
    if endpoint.is_claude:
        return await _claude_chat(endpoint, body, operation=operation, timeout=timeout, parse=parse)
    body = {"model": endpoint.model, **body}
    if body.get("tools") and body["model"] in _NO_REASONING_MODELS:
        body["reasoning_effort"] = "none"
    headers = {"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"}
    url = f"{endpoint.base_url}/chat/completions"
    result = Result()
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=body)
            for _ in range(4):
                adapted = _adapt(body, response.text.lower()) if response.status_code == 400 else None
                if adapted is None:
                    break
                body = adapted
                response = await client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            result.error = {"httpStatus": response.status_code, "body": response.text[:4000]}
            result.retryable = response.status_code >= 500 or response.status_code == 429
            logger.error("%s %s HTTP %s: %s", operation, endpoint.model, response.status_code, response.text[:400])
        else:
            result.data = response.json()
            result.value = parse(result.data)
    except Rejected as exc:
        result.error, result.retryable = exc.error, exc.retryable
        logger.error("%s %s answer rejected: %s", operation, endpoint.model, exc)
    except Exception as exc:  # network, timeout, bad JSON body
        result.error, result.retryable = {"reason": type(exc).__name__, "message": str(exc)}, True
        logger.error("%s %s error: %s: %s", operation, endpoint.model, type(exc).__name__, exc)
    input_tokens, cached, output_tokens = ai_log.chat_usage(result.data)
    logger.info("%s usage model=%s prompt=%s cached=%s completion=%s", operation, endpoint.model,
                input_tokens, cached, output_tokens)
    await ai_log.report(operation=operation, provider=ai_log.provider_of(endpoint.base_url), model=endpoint.model,
                        ok=result.ok, request=body, response=result.data, error=result.error,
                        input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens,
                        latency_ms=int((time.monotonic() - started) * 1000))
    return result


async def responses(endpoint: Endpoint, body: dict, *, operation: str, timeout: float,
                    client: Any = None) -> Result:
    """One OpenAI Responses call (web_search), reported with its search count. `client` reuses an
    open httpx client. A Claude endpoint runs the same search with Claude's web_search tool."""
    if endpoint.is_claude:
        return await _claude_web_search(endpoint, body, operation=operation, timeout=timeout)
    body = {"model": endpoint.model, **body}
    url = endpoint.base_url if endpoint.base_url.endswith("/responses") else f"{endpoint.base_url}/responses"
    headers = {"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"}
    result = Result()
    started = time.monotonic()
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=timeout) as own:
                response = await own.post(url, headers=headers, json=body)
        else:
            response = await client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            result.error = {"httpStatus": response.status_code, "body": response.text[:4000]}
            logger.warning("%s failed (%s): %s", operation, response.status_code, response.text[:400])
        else:
            data = response.json()
            if isinstance(data, dict):
                result.data = result.value = data
            else:
                result.error = {"reason": "response is not a JSON object"}
    except Exception as exc:
        result.error = {"reason": type(exc).__name__, "message": str(exc)}
        logger.warning("%s request failed: %s", operation, exc)
    input_tokens, cached, output_tokens, searches = ai_log.responses_usage(result.data)
    await ai_log.report(operation=operation, provider=ai_log.provider_of(endpoint.base_url), model=endpoint.model,
                        ok=result.ok, request=body, response=result.data, error=result.error,
                        input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens,
                        web_search_calls=searches, latency_ms=int((time.monotonic() - started) * 1000))
    return result


async def _claude_chat(endpoint: Endpoint, body: dict, *, operation: str, timeout: float,
                       parse: Callable[[dict], Any]) -> Result:
    params = anthropic_adapter.messages_params(endpoint.model, body)
    result = Result()
    raw: dict | None = None
    started = time.monotonic()
    try:
        raw = await anthropic_adapter.create(anthropic_adapter.client(endpoint.api_key, endpoint.base_url), params,
                                             timeout)
        result.data = anthropic_adapter.chat_completion(raw)
        result.value = parse(result.data)
    except Rejected as exc:
        result.error, result.retryable = exc.error, exc.retryable
        logger.error("%s %s answer rejected: %s", operation, endpoint.model, exc)
    except Exception as exc:  # SDK status errors, network, refusal
        result.error, result.retryable = anthropic_adapter.status_error(exc)
        logger.error("%s %s error: %s: %s", operation, endpoint.model, type(exc).__name__, exc)
    usage = _claude_log_usage(raw)
    logger.info("%s usage model=%s prompt=%s cached=%s cache_write=%s/%s completion=%s", operation,
                endpoint.model, usage.get("input_tokens"), usage.get("cached_input_tokens"),
                usage.get("cache_write_5m_tokens"), usage.get("cache_write_1h_tokens"), usage.get("output_tokens"))
    await ai_log.report(operation=operation, provider="anthropic", model=endpoint.model, ok=result.ok,
                        request=params, response=raw, error=result.error, **usage,
                        latency_ms=int((time.monotonic() - started) * 1000))
    return result


def _claude_log_usage(raw: Any) -> dict[str, int]:
    """`ai_log.report` token arguments read from the raw Messages answer(s), so cache writes keep
    their own (higher) price; empty when no answer came back."""
    usage = ai_log.anthropic_usage(raw)
    if usage is None:
        return {}
    return {"input_tokens": usage["input"], "cached_input_tokens": usage["cached"],
            "cache_write_5m_tokens": usage["cache_write_5m"], "cache_write_1h_tokens": usage["cache_write_1h"],
            "output_tokens": usage["output"]}


async def _claude_web_search(endpoint: Endpoint, body: dict, *, operation: str, timeout: float) -> Result:
    params = anthropic_adapter.web_search_params(endpoint.model, body)
    result = Result()
    raw: list[dict] = []
    started = time.monotonic()
    try:
        result.data, raw = await anthropic_adapter.web_search(
            anthropic_adapter.client(endpoint.api_key, endpoint.base_url), params, timeout)
        result.value = result.data
    except Exception as exc:
        result.error, _ = anthropic_adapter.status_error(exc)
        logger.warning("%s request failed: %s", operation, exc)
    billed = (ai_log.anthropic_usage(raw) or {}).get("searches") or 0
    # Claude reports the searches it bills; counting result blocks is the fallback.
    searches = billed or ai_log.responses_usage(result.data)[3]
    await ai_log.report(operation=operation, provider="anthropic", model=endpoint.model, ok=result.ok,
                        request=params, response=raw or None, error=result.error, **_claude_log_usage(raw),
                        web_search_calls=searches, latency_ms=int((time.monotonic() - started) * 1000))
    return result
