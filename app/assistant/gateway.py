"""Java callbacks for one assistant turn: steps, tools, the reply, failure and the call log."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class JavaGateway:
    """Talks to /v1/api/internal/assistant/turns/{turnId}/... with the internal token."""

    def __init__(self, callback_base_url: str, turn_id: str, token: str):
        self.root = callback_base_url.rstrip("/") + f"/v1/api/internal/assistant/turns/{turn_id}"
        self.headers = {"X-Internal-Token": token}

    async def _post(self, path: str, body: Any, timeout: float = 30) -> Any:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.root + path, headers=self.headers, json=body)
            response.raise_for_status()
            data = response.json() if response.content else {}
            return data.get("data") if isinstance(data, dict) else None

    async def step(self, key: str, params: dict | None = None, text: str | None = None) -> None:
        """Best effort: a lost step only makes the thinking view less detailed."""
        try:
            await self._post("/steps", {"key": key, "params": params or {}, "text": text}, timeout=10)
        except Exception as exc:
            logger.warning("assistant step %s not delivered: %s", key, exc)

    async def tool(self, name: str, arguments: dict) -> dict:
        try:
            data = await self._post(f"/tools/{name}", arguments, timeout=45)
            return data if isinstance(data, dict) else {"result": data}
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise TurnClosed() from exc
            logger.warning("assistant tool %s failed: %s", name, exc)
            return {"error": "The tool is unavailable right now."}
        except Exception as exc:
            logger.warning("assistant tool %s failed: %s", name, exc)
            return {"error": "The tool is unavailable right now."}

    async def reply(self, body: dict) -> None:
        await self._post("/reply", body, timeout=60)

    async def fail(self, error: str) -> None:
        try:
            await self._post("/fail", {"error": error[:1000]}, timeout=30)
        except Exception as exc:
            logger.error("assistant failure not delivered: %s", exc)

    async def ai_call(self, entry: dict) -> Any:
        """Logs one model call; Java answers with the asker's credit status."""
        return await self._post("/ai-calls", entry, timeout=10)


class TurnClosed(Exception):
    """Java says the turn is no longer running (timed out); stop working on it."""
