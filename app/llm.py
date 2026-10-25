"""A configured model tier with JSON output and an optional SQLite cache, for the trip planner."""
from __future__ import annotations
import asyncio, hashlib, json, logging, os, sqlite3
from pathlib import Path
from typing import Any
from . import providers
from .config import LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class CachedModel:
    """One configured model. `prefix` names the env triple <PREFIX>_BASE_URL/_API_KEY/_MODEL.

    If the primary prefix is not fully configured, `fallback_prefix` is used instead so a
    deployment with only CHEAP_LLM_* still works.
    """

    def __init__(self, prefix: str, fallback_prefix: str | None = None):
        self.endpoint = providers.from_tiers(*(p for p in (prefix, fallback_prefix) if p))
        self.prefix = self.endpoint.name
        logger.info("LLM %s -> %s model=%s configured=%s", prefix, self.prefix, self.model, self.configured)
        path = Path(os.getenv("LLM_CACHE_PATH", "/tmp/ai_trip_llm_cache.sqlite3"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    @property
    def url(self) -> str:
        return self.endpoint.base_url

    @property
    def key(self) -> str:
        return self.endpoint.api_key

    @property
    def model(self) -> str:
        return self.endpoint.model

    @property
    def configured(self) -> bool:
        return self.endpoint.configured

    async def json(self, system: str, payload: Any, *, max_tokens: int = 4096, temperature: float = 0.3,
                   cache: bool = True, retries: int = 1, operation: str = "CHAT",
                   cache_key: str | None = None, timeout: float | None = None,
                   reasoning_effort: str | None = None) -> Any | None:
        """Call the model and return the parsed JSON (object or array). None on failure.

        `cache_key` is the provider-side prompt cache routing key (OpenAI `prompt_cache_key`):
        requests sharing a static system prefix land on the same cache. Only sent to OpenAI.

        `reasoning_effort` bounds a thinking model's reasoning (Claude `output_config.effort`, OpenAI
        reasoning models); a host that rejects the field gets the request again without it."""
        if not self.configured:
            logger.error("LLM %s not configured", self.prefix)
            return None
        user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        key = hashlib.sha256(json.dumps({"m": self.model, "s": system, "u": user, "t": temperature},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if cache:
            row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if row:
                logger.info("LLM cache hit (%s)", self.prefix)
                return json.loads(row[0])
        body = {"temperature": temperature, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if cache_key and "openai.com" in self.url:
            body["prompt_cache_key"] = cache_key
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        logger.info("LLM call %s model=%s system=%d chars payload=%d chars", self.prefix, self.model, len(system), len(user))
        for attempt in range(retries + 1):
            result = await providers.chat(self.endpoint, body, operation=operation,
                                          timeout=timeout or LLM_TIMEOUT_SECONDS, parse=providers.json_answer)
            if result.ok or not result.retryable:
                break
            logger.warning("LLM attempt %d failed (%s)", attempt + 1, self.prefix)
            await asyncio.sleep(1.5)
        value = result.value if result.ok else None
        if value is not None and cache:
            self.db.execute("INSERT OR REPLACE INTO cache(key,value) VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
        return value
