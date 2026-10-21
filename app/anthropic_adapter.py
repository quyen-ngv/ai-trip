"""Claude through the official Anthropic SDK, behind `providers.chat` / `providers.responses`.

Callers keep speaking OpenAI's shapes (chat/completions messages and tools, a Responses body with
`web_search`); this module turns a request into a Messages API call and the answer back into the
OpenAI shape, so trip planning, Pengoo and social extraction run on Claude without their own code
knowing. What changes on the way:

- Leading system messages become `system`; later ones stay mid-conversation `role: "system"`
  messages; consecutive tool results are sent together in one user turn.
- `temperature` is dropped (current Claude models reject anything but the default), `max_tokens`
  gets room for adaptive thinking, and `reasoning_effort` becomes `output_config.effort`.
- `response_format` is dropped: every prompt that wants JSON says so, and the answer is parsed the
  same way as before.
- The assistant's own content blocks ride along as `_anthropic_content` and are sent back verbatim,
  because thinking blocks must return unmodified with tool results.
- `stop_reason: "refusal"` is reported as an unusable answer, not as text.
"""
from __future__ import annotations

import json
import re
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
# Thinking counts toward max_tokens: a cap sized for the answer alone (the 120-token topic
# pre-filter) would be spent on thinking before any text.
_MIN_MAX_TOKENS = 4096
_EFFORTS = {"none": "low", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
            "xhigh": "xhigh", "max": "max"}
_DATA_URL = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.S)
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "stop", "max_tokens": "length",
           "tool_use": "tool_calls"}
_MAX_CONTINUATIONS = 3
_clients: dict[tuple[str, str], anthropic.AsyncAnthropic] = {}


class Refused(Exception):
    """The model declined the request (`stop_reason: "refusal"`)."""

    def __init__(self, category: Any, explanation: Any):
        super().__init__(f"category={category} {explanation or ''}".strip())
        self.category, self.explanation = category, explanation


def client(api_key: str, base_url: str) -> anthropic.AsyncAnthropic:
    """One SDK client per key and host. The SDK wants the host without `/v1`."""
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/")) or DEFAULT_BASE_URL
    key = (api_key, root)
    if key not in _clients:
        _clients[key] = anthropic.AsyncAnthropic(api_key=api_key, base_url=root)
    return _clients[key]


async def create(endpoint_client: anthropic.AsyncAnthropic, params: dict, timeout: float) -> dict:
    """One Messages call, streamed so long answers and adaptive thinking never hit the SDK's
    non-streaming time limit. Returns the final message as a plain dict."""
    async with endpoint_client.with_options(timeout=timeout).messages.stream(**params) as stream:
        final = await stream.get_final_message()
    return final.to_dict()


# --- chat/completions -> Messages ------------------------------------------------------------

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text") or "" for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _image(url: str) -> dict:
    match = _DATA_URL.match(url or "")
    if match:
        return {"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _user_content(content: Any) -> list[dict] | str:
    if not isinstance(content, list):
        return _text(content)
    blocks: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            blocks.append(_image((part.get("image_url") or {}).get("url") or ""))
        elif part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def _assistant_content(message: dict) -> list[dict]:
    if message.get("_anthropic_content"):
        return message["_anthropic_content"]
    blocks: list[dict] = []
    if _text(message.get("content")).strip():
        blocks.append({"type": "text", "text": _text(message.get("content"))})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        blocks.append({"type": "tool_use", "id": call.get("id"), "name": function.get("name"), "input": arguments})
    return blocks


def _effort(value: Any) -> str | None:
    return _EFFORTS.get(str(value).strip().lower()) if value else None


def messages_params(model: str, body: dict) -> dict:
    """Messages API parameters for an OpenAI chat/completions body."""
    system: list[str] = []
    messages: list[dict] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        if role == "system" and not messages:
            system.append(_text(message.get("content")))
        elif role == "system":
            messages.append({"role": "system", "content": _text(message.get("content"))})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": _assistant_content(message)})
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": message.get("tool_call_id"),
                      "content": _text(message.get("content"))}
            if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) \
                    and messages[-1]["content"] and messages[-1]["content"][-1].get("type") == "tool_result":
                messages[-1]["content"].append(result)  # parallel results go back in one turn
            else:
                messages.append({"role": "user", "content": [result]})
        else:
            messages.append({"role": "user", "content": _user_content(message.get("content"))})
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or _MIN_MAX_TOKENS
    params: dict[str, Any] = {"model": model, "max_tokens": max(int(max_tokens), _MIN_MAX_TOKENS),
                              "messages": messages,
                              # Caches the longest stable prefix (system prompt, tool list, history).
                              "cache_control": {"type": "ephemeral"}}
    if system:
        params["system"] = "\n\n".join(system)
    effort = _effort(body.get("reasoning_effort"))
    if effort:
        params["output_config"] = {"effort": effort}
    tools = [tool.get("function") or {} for tool in body.get("tools") or [] if tool.get("type") == "function"]
    if tools:
        params["tools"] = [{"name": t.get("name"), "description": t.get("description") or "",
                            "input_schema": t.get("parameters") or {"type": "object", "properties": {}}}
                           for t in tools]
    return params


def chat_completion(message: dict) -> dict:
    """The OpenAI chat/completions shape of a Messages answer. Raises Refused on a refusal."""
    if message.get("stop_reason") == "refusal":
        details = message.get("stop_details") or {}
        raise Refused(details.get("category"), details.get("explanation"))
    content = message.get("content") or []
    text = "".join(block.get("text") or "" for block in content if block.get("type") == "text")
    tool_calls = [{"id": block.get("id"), "type": "function",
                   "function": {"name": block.get("name"),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}}
                  for block in content if block.get("type") == "tool_use"]
    reply: dict[str, Any] = {"role": "assistant", "content": text, "_anthropic_content": content}
    if tool_calls:
        reply["tool_calls"] = tool_calls
    usage = message.get("usage") or {}
    cached = usage.get("cache_read_input_tokens") or 0
    prompt = (usage.get("input_tokens") or 0) + cached + (usage.get("cache_creation_input_tokens") or 0)
    output = usage.get("output_tokens") or 0
    return {"model": message.get("model"),
            "choices": [{"message": reply, "finish_reason": _FINISH.get(message.get("stop_reason"), "stop")}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


# --- Responses web_search -> Messages web_search ------------------------------------------

def web_search_params(model: str, body: dict) -> dict:
    """Messages parameters for a Responses body that asks for `web_search`. Dynamic filtering
    (`web_search_20260209`) is not offered on Haiku, which gets the basic tool."""
    tool_type = "web_search_20250305" if "haiku" in model else "web_search_20260209"
    search: dict[str, Any] = {"type": tool_type, "name": "web_search"}
    if body.get("max_tool_calls"):
        search["max_uses"] = int(body["max_tool_calls"])
    params: dict[str, Any] = {"model": model, "max_tokens": max(int(body.get("max_output_tokens") or 0), 16000),
                              "messages": [{"role": "user", "content": _text(body.get("input"))}],
                              "tools": [search], "cache_control": {"type": "ephemeral"}}
    if body.get("instructions"):
        params["system"] = body["instructions"]
    effort = _effort((body.get("reasoning") or {}).get("effort"))
    if effort:
        params["output_config"] = {"effort": effort}
    return params


async def web_search(endpoint_client: anthropic.AsyncAnthropic, params: dict, timeout: float) -> tuple[dict, list[dict]]:
    """Runs the search turn to the end (`pause_turn` is resumed a few times). Returns the
    Responses-shaped body the callers parse, and the raw Messages answers for the call log."""
    raw: list[dict] = []
    messages = list(params["messages"])
    for _ in range(_MAX_CONTINUATIONS + 1):
        message = await create(endpoint_client, {**params, "messages": messages}, timeout)
        raw.append(message)
        if message.get("stop_reason") != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": message.get("content") or []}]
    final = raw[-1]
    if final.get("stop_reason") == "refusal":
        details = final.get("stop_details") or {}
        raise Refused(details.get("category"), details.get("explanation"))
    output: list[dict] = []
    text: list[str] = []
    input_tokens = cached = output_tokens = 0
    for message in raw:
        for block in message.get("content") or []:
            if block.get("type") == "web_search_tool_result" and isinstance(block.get("content"), list):
                sources = [{"url": r.get("url"), "title": r.get("title")} for r in block["content"]
                           if isinstance(r, dict) and r.get("url")]
                output.append({"type": "web_search_call", "action": {"sources": sources}})
        usage = message.get("usage") or {}
        cached += usage.get("cache_read_input_tokens") or 0
        input_tokens += (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0) \
            + (usage.get("cache_creation_input_tokens") or 0)
        output_tokens += usage.get("output_tokens") or 0
    # Citations split the answer into several text blocks; joined, they are the model's text.
    text.append("".join(block.get("text") or "" for block in final.get("content") or [] if block.get("type") == "text"))
    output.append({"type": "message", "content": [{"type": "output_text", "text": "".join(text)}]})
    return {"model": final.get("model"), "output": output,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "input_tokens_details": {"cached_tokens": cached}}}, raw


def status_error(exc: Exception) -> tuple[dict, bool]:
    """(error for the call log, whether an identical call could succeed) for an SDK exception."""
    if isinstance(exc, anthropic.APIStatusError):
        return ({"httpStatus": exc.status_code, "body": str(exc.message)[:4000]},
                exc.status_code >= 500 or exc.status_code in (408, 409, 429))
    if isinstance(exc, Refused):
        return {"reason": "refusal", "message": str(exc)}, False
    return {"reason": type(exc).__name__, "message": str(exc)}, True
