import json

import pytest

from app.assistant import graph as graph_module
from app.assistant.graph import build_graph, clean_reply, initial_messages, parse_final, run_turn
from app.assistant.tools import collect_names, step_for, tools_for

PLACE = "11111111-1111-4111-8111-111111111111"
INVENTED = "99999999-9999-4999-8999-999999999999"
ACTIVITY = "22222222-2222-4222-8222-222222222222"

TURN = {
    "turnId": "t1", "conversationId": "c1", "conversationType": "TRIP", "locale": "vi",
    "user": {"id": "u1", "name": "An"},
    "trip": {"id": "trip1", "name": "Đà Nẵng 3N", "days": [
        {"day": 2, "activities": [{"id": ACTIVITY, "name": "Khách sạn Mường Thanh", "lat": 16.06, "lng": 108.24}]}]},
    "history": [{"role": "user", "name": "Bình", "text": "Mai đi đâu?"}, {"role": "assistant", "text": "Đi biển nhé"}],
    "question": "tối ngày 2 ăn gì gần khách sạn?",
    "callbackBaseUrl": "http://java", "limits": {"maxToolCalls": 2, "maxOutputTokens": 800},
}


class FakeJava:
    def __init__(self):
        self.steps, self.tools, self.replies, self.failures = [], [], [], []

    async def step(self, key, params=None, text=None):
        self.steps.append((key, params or {}, text))

    async def tool(self, name, arguments):
        self.tools.append((name, arguments))
        return {"near": "Khách sạn Mường Thanh", "places": [
            {"id": PLACE, "name": "Bún chả cá Bà Lữ", "kind": "food", "rating": 4.5}]}

    async def reply(self, body):
        self.replies.append(body)

    async def fail(self, error):
        self.failures.append(error)

    async def ai_call(self, entry):
        pass


class ScriptedModel:
    """Returns the scripted assistant messages in order and records what it was sent."""
    configured = True

    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = []

    async def complete(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.messages:
            return None
        message = dict(self.messages.pop(0))
        message.setdefault("tool_calls", [])
        message["_usage"] = {"input": 100, "output": 20}
        return message


def tool_call(name, args, call_id="call1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


FINAL = json.dumps({
    "answer": "Thử **Bún chả cá Bà Lữ** nhé!",
    "places": [{"placeId": PLACE, "note": "Gần khách sạn"}, {"placeId": INVENTED, "note": "bịa"}],
    "proposals": [{"action": "ADD", "placeId": PLACE, "dayNumber": "2", "startTime": "19:00", "endTime": "25:00",
                   "summary": "Ăn tối ngày 2"}, {"action": "DELETE_ALL"}],
    "suggestions": ["Cà phê gần đó?", ""],
}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_agent_calls_a_tool_shows_steps_and_replies_with_only_real_places():
    java = FakeJava()
    model = ScriptedModel(
        {"content": "Mình xem quán gần khách sạn ngày 2 nhé.",
         "tool_calls": [tool_call("search_places", {"query": "bún chả cá", "activity_id": ACTIVITY})]},
        {"content": FINAL},
    )
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})

    assert java.tools == [("search_places", {"query": "bún chả cá", "activity_id": ACTIVITY})]
    keys = [s[0] for s in java.steps]
    assert keys == ["thought", "search_places", "writing_answer"]
    assert java.steps[0][2] == "Mình xem quán gần khách sạn ngày 2 nhé."
    assert java.steps[1][1] == {"query": "bún chả cá", "near": "Khách sạn Mường Thanh"}
    reply = java.replies[0]
    assert [p["placeId"] for p in reply["places"]] == [PLACE]          # invented id dropped
    assert reply["proposals"] == [{"action": "ADD", "placeId": PLACE, "dayNumber": 2, "startTime": "19:00",
                                   "summary": "Ăn tối ngày 2"}]          # bad time and bad action dropped
    assert reply["suggestions"] == ["Cà phê gần đó?"]
    assert reply["inputTokens"] == 200 and reply["outputTokens"] == 40
    # second round saw the tool result
    assert any(m.get("role") == "tool" for m in model.calls[1]["messages"])


@pytest.mark.asyncio
async def test_tool_budget_forces_an_answer():
    java = FakeJava()
    looping = {"content": "", "tool_calls": [tool_call("search_places", {"query": "x"})]}
    model = ScriptedModel(looping, looping, {"content": '{"answer": "Xong"}'})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert len(java.tools) == 2
    assert model.calls[2]["tools"] is None
    assert java.replies[0]["markdown"] == "Xong"


@pytest.mark.asyncio
async def test_run_turn_reports_failure_once(monkeypatch):
    java = FakeJava()
    monkeypatch.setattr(graph_module, "JavaGateway", lambda *args: java)
    await run_turn(TURN, "token", model=ScriptedModel())   # model returns None at once
    assert java.replies == [] and len(java.failures) == 1
    assert java.steps[0][0] == "reading_trip"


def test_final_answer_parsing_tolerates_fences_and_plain_text():
    assert parse_final('```json\n{"answer": "Hi"}\n```')["answer"] == "Hi"
    assert parse_final('Sure! {"answer": "Hi", "places": []} done')["answer"] == "Hi"
    assert parse_final("Just text") == {"answer": "Just text"}


def test_clean_reply_bounds_everything():
    reply = clean_reply({"answer": "x" * 9000, "suggestions": [str(i) for i in range(9)]}, set(), {})
    assert len(reply["markdown"]) == 8000 and len(reply["suggestions"]) == 4


def test_messages_put_static_rules_first_and_name_people():
    messages = initial_messages(TURN)
    assert messages[0]["role"] == "system" and "Pengoo" in messages[0]["content"]
    assert "Vietnamese" in messages[1]["content"] and ACTIVITY in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "Bình: Mai đi đâu?"}
    assert messages[3] == {"role": "assistant", "content": "Đi biển nhé"}
    assert messages[-1]["content"] == "An: tối ngày 2 ăn gì gần khách sạn?"


def test_direct_chat_gets_trip_tools_and_names_resolve():
    assert {t["function"]["name"] for t in tools_for("ASSISTANT")} >= {"list_my_trips", "get_trip"}
    assert "list_my_trips" not in {t["function"]["name"] for t in tools_for("TRIP")}
    names, places = {}, set()
    collect_names({"places": [{"id": PLACE, "name": "Quán A", "kind": "food"}], "trips": [{"id": "t9", "name": "Huế"}]},
                  names, places)
    assert places == {PLACE} and names["t9"] == "Huế"
    assert step_for("get_trip", {"trip_id": "t9"}, names) == ("trip_details", {"name": "Huế"})


@pytest.mark.asyncio
async def test_reasoning_model_with_tools_retries_with_reasoning_off(monkeypatch):
    import httpx
    from app.assistant import chat_model

    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("reasoning_effort") != "none":
            return httpx.Response(400, json={"error": {"message": "Function tools with reasoning_effort are not "
                                                                  "supported. Set reasoning_effort to 'none'."}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(chat_model.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(chat_model, "_NO_REASONING_MODELS", set())
    monkeypatch.setattr(chat_model.ai_log, "report", lambda **kw: _noop())
    for key, value in {"BASE_URL": "https://llm.test/v1", "API_KEY": "k", "MODEL": "gpt-test"}.items():
        monkeypatch.setenv(f"ASSISTANT_LLM_{key}", value)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]

    first = await chat_model.ChatModel().complete([{"role": "user", "content": "hi"}], tools)
    second = await chat_model.ChatModel().complete([{"role": "user", "content": "hi"}], tools)

    assert first["content"] == "ok" and second["content"] == "ok"
    assert [b.get("reasoning_effort") for b in sent] == [None, "none", "none"]


async def _noop():
    return None
