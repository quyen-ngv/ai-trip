"""Pengoo reads one web page: a link the user sent, or a result web_search found.

Search results are short snippets. For "where to eat X" they are mostly list articles ("Top 10
quán bánh tráng cuốn thịt heo…") whose snippets hold no shop names, so Pengoo used to answer with
bare links. Reading the best page gives it the names and addresses.

Offered with web_search (same switch, same unverified status: text and a source link, never cards
or proposals). Plain HTTP: pages built by JavaScript (Facebook, TikTok, Google Maps) come back
nearly empty, and the model is told so.

Only public http(s) hosts are fetched: every hop of a redirect is resolved and refused when any
address is private, loopback, link-local or reserved, so a link cannot reach the worker's own
network.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from .web_search import _cache_store
from ..research_cache import ResearchCache

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 12
MAX_BYTES = 1_500_000
MAX_REDIRECTS = 4
TEXT_CHARS = 6000
CACHE_TTL_SECONDS = 24 * 3600
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/129.0 Mobile Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    "Accept-Language": "vi,en;q=0.8",
}
_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "form", "nav", "footer", "aside"}
_BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "dl", "dd", "dt",
          "blockquote", "figcaption", "header"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """Visible text with structure kept: headings as '## ', list items as '- ', blocks on lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self.parts.append("\n\n## ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _HEADINGS or tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = [_SPACES.sub(" ", line).strip() for line in "".join(self.parts).split("\n")]
        joined = "\n".join(line for line in lines if line not in ("", "-", "##"))
        return _BLANK_LINES.sub("\n\n", joined).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """(title, text) of an HTML page."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # broken markup: keep what was read
        pass
    return " ".join(parser.title.split()), parser.text()


async def _public_host(host: str, port: int) -> bool:
    """True when every address the host resolves to is a public one."""
    if not host:
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    return bool(addresses) and all(ipaddress.ip_address(address.split("%")[0]).is_global
                                   for address in addresses)


def _checked_url(url: str) -> str | None:
    parts = urlsplit((url or "").strip())
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return None
    return parts.geturl()


async def read(url: str) -> dict[str, Any]:
    """{"url", "title", "text", "note"} or {"error": ...}; never raises."""
    current = _checked_url(url)
    if current is None:
        return {"error": "Only a full http(s) link can be read."}
    first = urlsplit(current)
    if not await _public_host(first.hostname or "", first.port or (443 if first.scheme == "https" else 80)):
        return {"error": "That address cannot be opened."}
    store = _cache_store()
    cache_key = ResearchCache.key("assistant-read-url-v1", {"url": current})
    cached = store.get(cache_key) if store else None
    if isinstance(cached, dict) and cached.get("text"):
        return cached
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                parts = urlsplit(current)
                port = parts.port or (443 if parts.scheme == "https" else 80)
                if not await _public_host(parts.hostname or "", port):
                    return {"error": "That address cannot be opened."}
                async with client.stream("GET", current) as response:
                    if response.is_redirect and response.headers.get("location"):
                        following = _checked_url(urljoin(current, response.headers["location"]))
                        if following is None:
                            return {"error": "The page redirects somewhere that cannot be opened."}
                        current = following
                        continue
                    if response.status_code >= 400:
                        return {"error": f"The page answered {response.status_code}."}
                    kind = response.headers.get("content-type", "").lower()
                    if kind and "html" not in kind and "text/plain" not in kind:
                        return {"error": "That link is not a web page (it is " + kind.split(";")[0] + ")."}
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) >= MAX_BYTES:
                            break
                    content = bytes(body).decode(response.charset_encoding or "utf-8", errors="replace")
                    break
            else:
                return {"error": "Too many redirects."}
    except Exception as exc:  # network, TLS, timeout
        logger.warning("assistant read_url failed for %s: %s", current, exc)
        return {"error": "The page could not be opened right now."}
    title, text = html_to_text(content) if "<" in content[:2000] else ("", content.strip())
    if len(text) < 200:
        return {"url": current, "title": title, "text": text,
                "note": "Almost no text: the page is probably built by JavaScript or blocks readers. "
                        "Try another result."}
    result = {"url": current, "title": title, "text": text[:TEXT_CHARS] + ("…" if len(text) > TEXT_CHARS else ""),
              "note": "Unverified web page. Name what it says in the answer with this link as the source; "
                      "no cards, no proposals."}
    if store:
        store.put(cache_key, result, CACHE_TTL_SECONDS)
    logger.info("assistant read_url: %s (%d chars)", current, len(text))
    return result
