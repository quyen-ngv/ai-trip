"""Thin OpenAI-compatible chat client with JSON output and an optional SQLite cache."""
from __future__ import annotations
import asyncio, hashlib, json, logging, os, re, sqlite3
from pathlib import Path
from typing import Any
import httpx
from .config import LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


class CachedModel:
    """One configured model. `prefix` names the env triple <PREFIX>_BASE_URL/_API_KEY/_MODEL.

    If the primary prefix is not fully configured, `fallback_prefix` is used instead so a
    deployment with only CHEAP_LLM_* still works.
    """

    def __init__(self, prefix: str, fallback_prefix: str | None = None):
        chosen = prefix
        if not self._configured(prefix) and fallback_prefix and self._configured(fallback_prefix):
            chosen = fallback_prefix
        self.prefix = chosen
        self.url = os.getenv(f"{chosen}_BASE_URL", "").rstrip("/")
        self.key = os.getenv(f"{chosen}_API_KEY", "")
        self.model = os.getenv(f"{chosen}_MODEL", "")
        logger.info("LLM %s -> %s model=%s configured=%s", prefix, chosen, self.model, self.configured)
        path = Path(os.getenv("LLM_CACHE_PATH", "/tmp/ai_trip_llm_cache.sqlite3"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    @staticmethod
    def _configured(prefix: str) -> bool:
        return all(os.getenv(f"{prefix}_{k}", "") for k in ("BASE_URL", "API_KEY", "MODEL"))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key and self.model)

    async def json(self, system: str, payload: Any, *, max_tokens: int = 4096, temperature: float = 0.3,
                   cache: bool = True, retries: int = 1) -> Any | None:
        """Call the model and return the parsed JSON (object or array). None on failure."""
        if not self.configured:
            logger.error("LLM %s not configured", self.prefix)
            return None
        user = json.dumps(payload, ensure_ascii=False)
        key = hashlib.sha256(json.dumps({"m": self.model, "s": system, "u": user, "t": temperature},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if cache:
            row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if row:
                logger.info("LLM cache hit (%s)", self.prefix)
                return json.loads(row[0])
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        logger.info("LLM call %s model=%s system=%d chars payload=%d chars", self.prefix, self.model, len(system), len(user))
        value = None
        for attempt in range(retries + 1):
            value = await self._call(body)
            if value is not None:
                break
            logger.warning("LLM attempt %d failed (%s)", attempt + 1, self.prefix)
            await asyncio.sleep(1.5)
        if value is not None and cache:
            self.db.execute("INSERT OR REPLACE INTO cache(key,value) VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
        return value

    async def _call(self, body: dict) -> Any | None:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        content = ""
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
                r = await client.post(f"{self.url}/chat/completions", headers=headers, json=body)
                # Provider quirks: newer OpenAI models reject `max_tokens` (want `max_completion_tokens`)
                # and non-default `temperature`; some OpenAI-compatible hosts reject `response_format`.
                for _ in range(3):
                    if r.status_code != 400:
                        break
                    err = r.text.lower()
                    if "max_tokens" in err and "max_tokens" in body:
                        max_tokens = body.pop("max_tokens")
                        body["max_completion_tokens"] = max_tokens
                    elif "temperature" in err and "temperature" in body:
                        body = {k: v for k, v in body.items() if k != "temperature"}
                    elif "response_format" in err and "response_format" in body:
                        body = {k: v for k, v in body.items() if k != "response_format"}
                    else:
                        break
                    r = await client.post(f"{self.url}/chat/completions", headers=headers, json=body)
                if r.status_code != 200:
                    logger.error("LLM HTTP %s: %s", r.status_code, r.text[:400])
                    return None
                data = r.json()
            choices = data.get("choices") or []
            if not choices:
                logger.error("LLM returned no choices: %s", str(data)[:400])
                return None
            finish = choices[0].get("finish_reason")
            content = (choices[0].get("message") or {}).get("content") or ""
            if finish == "length":
                logger.error("LLM output truncated at max_tokens=%s",
                             body.get("max_completion_tokens", body.get("max_tokens")))
                return None
            if not content.strip():
                logger.error("LLM returned empty content (finish=%s)", finish)
                return None
            return json.loads(_FENCE.sub("", content))
        except json.JSONDecodeError as e:
            logger.error("LLM JSON parse error: %s | head=%r", e, content[:300])
        except Exception as e:  # network, timeout
            logger.error("LLM error: %s: %s", type(e).__name__, e)
        return None
