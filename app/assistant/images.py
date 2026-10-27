"""Photos sent with a question (a receipt, a ticket, a page of notes), made ready for the model.

Java sends the storage URLs of the question's photos (never hidden ones). They are downloaded here
and passed inline as base64 data URLs, so the model provider never has to reach our storage, and
the call log's redaction of data URLs (ai_log) keeps them out of the logs. A photo that cannot be
fetched is left out; one too large to inline goes as its URL.
"""
from __future__ import annotations

import asyncio
import base64
import logging

import httpx

logger = logging.getLogger(__name__)

MAX_IMAGES = 4
# Claude takes at most 5 MB per image; base64 adds a third, so inline only what stays under it.
MAX_INLINE_BYTES = 3_700_000
MAX_FETCH_BYTES = 12_000_000
FETCH_TIMEOUT_SECONDS = 10.0
_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


async def _one(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("assistant photo not loaded (%s): %s", type(exc).__name__, url[:120])
        return None
    if len(data) > MAX_FETCH_BYTES:
        return None
    mime = _sniff(data)
    if mime not in _TYPES:
        return None
    if len(data) > MAX_INLINE_BYTES:
        return url
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


async def load(urls: list) -> list[str]:
    """The photos as image URLs for the model (data URLs where possible), in order."""
    wanted = [str(u) for u in urls or [] if isinstance(u, str) and u.startswith(("http://", "https://"))][:MAX_IMAGES]
    if not wanted:
        return []
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        loaded = await asyncio.gather(*(_one(client, url) for url in wanted))
    return [image for image in loaded if image]
