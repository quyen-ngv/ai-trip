"""Pengoo's last-resort internet search, used when the catalogue has nothing.

Offered only when Java allows it for the turn (`limits.webSearch`, config ASSISTANT.WEB_SEARCH_ENABLED)
and a provider is configured here; the graph allows one call per turn. Results are short snippets
with their links. They never become place cards or proposals, because nothing in them is checked
against the catalogue.

Providers (ASSISTANT_WEB_SEARCH_PROVIDER):
  openai (default)  The OpenAI Responses API with its built-in web_search tool, the same way the
                    trip planner researches (`research.py`). Uses ASSISTANT_WEB_SEARCH_BASE_URL /
                    _API_KEY / _MODEL, falling back to CHEAP_LLM_*. Logged to ai_api_calls.
  tinyfish          TinyFish Search (free up to 12,000 a day, 30 a minute); title, url and snippet
                    per result, no model call. Needs ASSISTANT_WEB_SEARCH_API_KEY.
  tavily | brave    A search API; needs ASSISTANT_WEB_SEARCH_API_KEY.

A search API that fails (rate limit, outage) hands the query to OpenAI when CHEAP_LLM_* is set, so
the traveller still gets an answer; the ASSISTANT_WEB_SEARCH_* values then belong to that API.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

from .. import ai_log, config
from ..research import _output_text, _response_sources
from ..research_cache import ResearchCache

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
SNIPPET_CHARS = 300
TIMEOUT_SECONDS = 30
_TAG = re.compile(r"<[^>]+>")
# One search call per question: the second call used to open a page the first had already
# summarised, and OpenAI bills every call. Results are kept for a while: ticket prices, opening
# hours and event dates do not change between two travellers asking the same thing.
MAX_TOOL_CALLS = max(1, int(os.getenv("ASSISTANT_WEB_SEARCH_MAX_TOOL_CALLS", "1") or 1))
CACHE_TTL_SECONDS = max(0, int(os.getenv("ASSISTANT_WEB_SEARCH_CACHE_TTL_SECONDS", str(3 * 24 * 3600)) or 0))
_cache: ResearchCache | None = None
_cache_broken = False


def _cache_store() -> ResearchCache | None:
    global _cache, _cache_broken
    if CACHE_TTL_SECONDS <= 0 or _cache_broken:
        return None
    if _cache is None:
        try:
            _cache = ResearchCache(config.RESEARCH_CACHE_PATH)
        except Exception as exc:  # unwritable path: search still works, uncached
            logger.warning("assistant web search cache unavailable: %s", exc)
            _cache_broken = True
    return _cache

_INSTRUCTIONS = """Search the web for the query and answer for a traveller in Vietnam. Return JSON only:
{"results": [{"title": "...", "url": "https://...", "snippet": "the relevant fact in one or two sentences"}]}
Use at most 5 results, only from pages you actually read, prefer recent and local sources, and keep
each snippet factual (names, dates, address, price). No results → {"results": []}."""


def _provider() -> str:
    return os.getenv("ASSISTANT_WEB_SEARCH_PROVIDER", "openai").strip().lower() or "openai"


def _openai_settings() -> tuple[str, str, str]:
    # Behind another provider OpenAI is only the fallback, and ASSISTANT_WEB_SEARCH_* hold that
    # provider's key, so it runs on the planner's CHEAP_LLM_* settings.
    own = _provider() == "openai"

    def pick(key: str) -> str:
        value = os.getenv(f"ASSISTANT_WEB_SEARCH_{key}", "").strip() if own else ""
        return value or os.getenv(f"CHEAP_LLM_{key}", "").strip()
    return pick("BASE_URL").rstrip("/"), pick("API_KEY"), pick("MODEL")


def configured() -> bool:
    if _provider() == "openai":
        return all(_openai_settings())
    return bool(os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip())


async def search(query: str) -> dict[str, Any]:
    """{"results": [{title, url, snippet}], "note": ...} or {"error": ...}; never raises."""
    query = (query or "").strip()[:200]
    if not query:
        return {"error": "query is required"}
    if not configured():
        return {"error": "Web search is not available."}
    provider = _provider()
    store = _cache_store()
    cache_key = ResearchCache.key("assistant-web-search-v1", {"provider": provider, "query": query.casefold()})
    cached = store.get(cache_key) if store else None
    if isinstance(cached, list) and cached:
        logger.info("assistant web search via %s: %d cached results", provider, len(cached))
        return _reply(cached)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            try:
                results = await _SEARCHERS.get(provider, _openai)(client, query)
            except Exception as exc:
                if provider == "openai" or not all(_openai_settings()):
                    raise
                logger.warning("assistant web search failed (%s), asking openai instead: %s", provider, exc)
                results = await _openai(client, query)
    except Exception as exc:  # network, quota, bad JSON
        logger.warning("assistant web search failed (%s): %s", provider, exc)
        return {"error": "Web search is unavailable right now."}
    logger.info("assistant web search via %s: %d results", provider, len(results))
    if store and results:
        store.put(cache_key, results, CACHE_TTL_SECONDS)
    return _reply(results)


def _reply(results: list[dict]) -> dict[str, Any]:
    return {"results": results,
            "note": "Unverified web snippets. Mention them as 'theo <source>' with the link; no cards, no proposals."}


async def _openai(client: httpx.AsyncClient, query: str) -> list[dict]:
    base_url, key, model = _openai_settings()
    endpoint = base_url if base_url.endswith("/responses") else f"{base_url}/responses"
    body = {"model": model, "instructions": _INSTRUCTIONS, "input": query,
            "tools": [{"type": "web_search", "search_context_size": "low"}], "tool_choice": "required",
            "include": ["web_search_call.action.sources"], "max_tool_calls": MAX_TOOL_CALLS, "store": False,
            "reasoning": {"effort": "low"}}
    started = time.monotonic()
    value: Any = None
    error: Any = None
    try:
        response = await client.post(endpoint, headers={"Authorization": f"Bearer {key}"}, json=body)
        if response.status_code != 200:
            error = {"httpStatus": response.status_code, "body": response.text[:4000]}
            raise RuntimeError(f"HTTP {response.status_code}")
        value = response.json()
    finally:
        input_tokens, cached, output_tokens, searches = ai_log.responses_usage(value)
        await ai_log.report(operation="WEB_SEARCH", provider=ai_log.provider_of(base_url), model=model,
                            ok=value is not None, request=body, response=value, error=error,
                            input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens,
                            web_search_calls=searches, latency_ms=int((time.monotonic() - started) * 1000))
    cited = _response_sources(value)
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", _output_text(value) or "").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {}
    rows = [_row(item.get("title"), item.get("url"), item.get("snippet"))
            for item in (parsed.get("results") if isinstance(parsed, dict) else None) or [] if isinstance(item, dict)]
    # Only links the search really returned; a model may otherwise make one up.
    rows = [row for row in rows if not cited or row["url"] in cited]
    if not rows and cited:
        rows = [_row(title, url, "") for url, title in list(cited.items())[:MAX_RESULTS]]
    return rows[:MAX_RESULTS]


async def _tavily(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    response = await client.post("https://api.tavily.com/search",
                                 headers={"Authorization": f"Bearer {key}"},
                                 json={"query": query, "max_results": MAX_RESULTS, "search_depth": "basic",
                                       "include_answer": False})
    response.raise_for_status()
    return [_row(item.get("title"), item.get("url"), item.get("content"))
            for item in (response.json().get("results") or [])][:MAX_RESULTS]


async def _brave(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    response = await client.get("https://api.search.brave.com/res/v1/web/search",
                                headers={"X-Subscription-Token": key, "Accept": "application/json"},
                                params={"q": query, "count": MAX_RESULTS})
    response.raise_for_status()
    items = ((response.json().get("web") or {}).get("results")) or []
    return [_row(item.get("title"), item.get("url"), item.get("description")) for item in items][:MAX_RESULTS]


async def _tinyfish(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    params = {"query": query}
    started = time.monotonic()
    response = await client.get("https://api.search.tinyfish.ai", headers={"X-API-Key": key}, params=params)
    ok = response.status_code == 200
    value = response.json() if ok else None
    # Logged so the admin AI-calls page shows these searches too; no tokens, and free, so no
    # web_search_calls count that the price table would flag as unpriced.
    await ai_log.report(operation="WEB_SEARCH", provider="tinyfish", model="search", ok=ok, request=params,
                        response=value, error=None if ok else {"httpStatus": response.status_code,
                                                               "body": response.text[:4000]},
                        latency_ms=int((time.monotonic() - started) * 1000))
    response.raise_for_status()
    items = (value or {}).get("results") or []
    return [_row(item.get("title"), item.get("url"), item.get("snippet"))
            for item in items if isinstance(item, dict) and item.get("url")][:MAX_RESULTS]


_SEARCHERS = {"tinyfish": _tinyfish, "tavily": _tavily, "brave": _brave}


def _row(title: Any, url: Any, snippet: Any) -> dict:
    text = " ".join(_TAG.sub("", str(snippet or "")).split())
    return {"title": str(title or "")[:120], "url": str(url or ""),
            "snippet": text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")}
