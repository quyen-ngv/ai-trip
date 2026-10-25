import json

import pytest

from app import ai_log
from app.assistant import graph as graph_module
from app.assistant.graph import build_graph, clean_cards, clean_destinations, clean_reply, initial_messages, parse_final, run_turn
from app.assistant.prompts import history_messages
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
    model = ScriptedModel({"content": "", "tool_calls": [tool_call("search_places", {"query": "x"})]},
                          {"content": "", "tool_calls": [tool_call("search_places", {"query": "y"})]},
                          {"content": '{"answer": "Xong"}'})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert len(java.tools) == 2
    # The same tool list every round: Claude refuses a thinking block whose tool list changed.
    assert model.calls[2]["tools"] == model.calls[0]["tools"]
    told = [m for m in model.calls[2]["messages"] if m.get("content") == graph_module.BUDGET_USED]
    assert len(told) == 1
    assert java.replies[0]["markdown"] == "Xong"


@pytest.mark.asyncio
async def test_calls_past_the_budget_are_refused_and_the_turn_still_ends():
    java = FakeJava()
    rounds = [{"content": "", "tool_calls": [tool_call("search_places", {"query": f"x{i}"})]} for i in range(4)]
    last = {"content": '{"answer": "Xong"}', "tool_calls": [tool_call("search_places", {"query": "y"})]}
    model = ScriptedModel(*rounds, last)
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert len(java.tools) == 2                       # nothing ran past the budget of 2
    assert len(model.calls) == 5
    # The budget notice was added once and stayed in place: the history only ever grows.
    for earlier, later in zip(model.calls, model.calls[1:]):
        assert later["messages"][:len(earlier["messages"])] == earlier["messages"]
    assert java.replies[0]["markdown"] == "Xong"


@pytest.mark.asyncio
async def test_a_repeated_call_is_answered_without_running_and_a_batch_is_one_tool_call():
    java = FakeJava()
    turn = {**TURN, "limits": {"maxToolCalls": 3, "maxOutputTokens": 800}}
    model = ScriptedModel(
        {"content": "", "tool_calls": [tool_call("search_places", {"query": "bún chả"}, "c1"),
                                       tool_call("search_places", {"query": "bún chả"}, "c2"),
                                       tool_call("noop", {}, "c3")]},
        {"content": "", "tool_calls": [tool_call("get_place_details", {"place_ids": [PLACE, "p2", PLACE]}, "c4"),
                                       tool_call("travel_time", {"legs": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]}, "c5")]},
        {"content": '{"answer": "Xong"}'})
    graph = build_graph(turn, java, model)
    await graph.ainvoke({"messages": initial_messages(turn), "tool_calls": 0})
    assert java.tools == [("search_places", {"query": "bún chả"}),
                          ("get_place_details", {"place_id": PLACE}), ("get_place_details", {"place_id": "p2"}),
                          ("travel_time", {"from": "a", "to": "b"}), ("travel_time", {"from": "b", "to": "c"})]
    first_round = [json.loads(m["content"]) for m in model.calls[1]["messages"] if m.get("role") == "tool"]
    assert first_round[1] == {"note": graph_module.REPEATED_CALL}
    assert "Unknown tool" in first_round[2]["error"]
    second_round = [json.loads(m["content"]) for m in model.calls[2]["messages"] if m.get("role") == "tool"][3:]
    assert len(second_round[0]["places"]) == 2 and [leg["from"] for leg in second_round[1]["legs"]] == ["a", "b"]
    assert java.replies[0]["markdown"] == "Xong"     # three real calls fitted the budget of 3


@pytest.mark.asyncio
async def test_a_tool_past_its_own_limit_is_refused_without_spending_the_budget():
    java = FakeJava()
    weather = [{"content": "", "tool_calls": [tool_call("get_weather", {"city": c}, c)]} for c in ("Huế", "Hội An")]
    model = ScriptedModel(*weather, {"content": '{"answer": "Xong"}'})
    graph = build_graph({**TURN, "limits": {"maxToolCalls": 5}}, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert [name for name, _ in java.tools] == ["get_weather"]
    refused = json.loads([m for m in model.calls[2]["messages"] if m.get("role") == "tool"][-1]["content"])
    assert "limit 1" in refused["error"]


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


def test_clean_reply_drops_a_copied_history_card_note():
    answer = "Thẻ địa điểm ở dưới nhé.\n[place cards shown: My Khe Beach (ac1c24ff-7e2c)]"
    assert clean_reply({"answer": answer}, set(), {})["markdown"] == "Thẻ địa điểm ở dưới nhé."


def test_clean_reply_keeps_an_extension_first_so_the_cap_never_drops_it():
    adds = [{"action": "ADD", "placeId": f"p{i}", "dayNumber": 4} for i in range(6)]
    reply = clean_reply({"answer": "ok", "proposals": adds + [{"action": "ADD_DAYS", "toDayNumber": "4"}]}, set(), {})
    assert len(reply["proposals"]) == 6
    assert reply["proposals"][0] == {"action": "ADD_DAYS", "toDayNumber": 4}


def test_clean_reply_keeps_day_change_fields_and_drops_bad_ones():
    reply = clean_reply({"answer": "ok", "proposals": [
        {"action": "SHIFT_DATES", "startDate": "2026-11-08"},
        {"action": "SHIFT_DATES", "startDate": "next week"},
        {"action": "PREPEND_DAYS", "days": "2"},
        {"action": "REPLACE", "activityId": "a1", "placeId": "p2", "startTime": "7pm"},
        {"action": "RENAME_TRIP", "summary": "unknown actions are dropped"}]}, set(), {})
    assert reply["proposals"][:3] == [{"action": "SHIFT_DATES", "startDate": "2026-11-08"},
                                      {"action": "SHIFT_DATES"}, {"action": "PREPEND_DAYS", "days": 2}]
    assert reply["proposals"][3] == {"action": "REPLACE", "activityId": "a1", "placeId": "p2"}
    assert len(reply["proposals"]) == 4


def test_a_place_shared_by_a_member_is_named_in_the_history():
    turn = {**TURN, "history": [{"role": "user", "name": "Bình", "text": "[shared place p9]",
                                 "places": [{"id": "p9", "name": "Chợ Hàn"}]}]}
    messages = history_messages(turn)
    assert messages[0]["content"] == "Bình: [shared place p9] [place: Chợ Hàn (p9)]"


def test_budget_tool_needs_a_trip_only_in_the_direct_chat():
    trip_budget = next(t for t in tools_for("TRIP") if t["function"]["name"] == "get_budget")
    direct_budget = [t for t in tools_for("ASSISTANT") if t["function"]["name"] == "get_budget"]
    assert trip_budget["function"]["parameters"]["required"] == []
    assert len(direct_budget) == 1 and direct_budget[0]["function"]["parameters"]["required"] == ["trip_id"]


def test_messages_put_static_rules_first_history_next_and_context_with_the_question_last():
    messages = initial_messages(TURN)
    assert messages[0]["role"] == "system" and "Pengoo" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Bình: Mai đi đâu?"}
    assert messages[2] == {"role": "assistant", "content": "Đi biển nhé"}
    last = messages[-1]
    assert last["role"] == "user" and len(messages) == 4
    # Per-turn context travels after the history so the history stays a cacheable prefix.
    assert last["content"].startswith("<context>\n") and "Reply in: Vietnamese." in last["content"]
    assert ACTIVITY in last["content"] and "lat" not in last["content"]
    assert last["content"].endswith("</context>\n\nAn: tối ngày 2 ăn gì gần khách sạn?")


def test_reply_language_follows_the_question_then_the_app():
    from app.assistant.prompts import reply_language
    assert reply_language("What should we eat today?", "vi") == "English"
    assert reply_language("Hà nội có quán phở nào kiểu local ko", "en") == "Vietnamese"
    assert reply_language("tim quan pho ngon o ha noi", "en") == "Vietnamese"
    assert reply_language("Hội An?", "en") == "English"          # a place name, not a Vietnamese question
    assert reply_language("오늘 뭐 먹지?", "vi") == "Korean"
    assert reply_language("Ok", "ja") == "Japanese"


def test_trip_snapshot_is_compacted_with_ride_minutes():
    from app.assistant.prompts import compact_trip
    trip = {"startDate": [2026, 12, 19], "today": {"date": [2026, 10, 24], "startsInDays": 56},
            "days": [{"day": 1, "date": [2026, 12, 19], "activities": [
                {"id": "a", "name": "Thác", "lat": 20.8129, "lng": 104.5916, "placeId": "p1"},
                {"id": "b", "name": "Cafe", "lat": 20.8206, "lng": 104.5995,
                 "note": "Gợi ý từ nguồn Web đã trích dẫn, chưa có trong dữ liệu của chúng tôi. Sources: h…"}]}]}
    out = compact_trip(trip)
    assert out["startDate"] == "2026-12-19" and out["today"]["date"] == "2026-10-24"
    first, second = out["days"][0]["activities"]
    assert out["days"][0]["date"] == "2026-12-19"
    assert "lat" not in first and first["rideMinToNext"] == 11
    assert second == {"id": "b", "name": "Cafe", "fromWeb": True}


def test_direct_chat_gets_trip_tools_and_names_resolve():
    assert {t["function"]["name"] for t in tools_for("ASSISTANT")} >= {"list_my_trips", "get_trip"}
    assert "list_my_trips" not in {t["function"]["name"] for t in tools_for("TRIP")}
    names = {}
    ids = {"place": set(), "event": set(), "meetup": set(), "trip": set()}
    collect_names({"places": [{"id": PLACE, "name": "Quán A", "kind": "food"}],
                   "trips": [{"id": "t9", "name": "Huế", "startDate": "2026-10-16"}]}, names, ids)
    assert ids["place"] == {PLACE} and ids["trip"] == {"t9"} and names["t9"] == "Huế"
    assert step_for("get_trip", {"trip_id": "t9"}, names) == ("trip_details", {"name": "Huế"})


def test_community_results_feed_cards_but_public_trips_never_become_open_buttons():
    names = {}
    ids = {"place": set(), "event": set(), "meetup": set(), "trip": set()}
    collect_names({"trips": [{"id": "pt1", "name": "Đà Lạt 3 ngày", "kind": "trip"}],
                   "latest": [{"id": "c1", "name": "Chợ đêm", "kind": "checkin", "placeId": "p7"}],
                   "dishes": [{"id": "d1", "name": "Bánh căn", "kind": "dish", "citySlug": "da-lat"}]}, names, ids)
    assert ids["public_trip"] == {"pt1"} and ids["trip"] == set()
    assert ids["checkin"] == {"c1"} and ids["dish"] == {"d1"} and "p7" in ids["place"]
    cards = clean_cards([{"type": "trip", "id": "pt1"}, {"type": "TRIP", "id": "pt1"}, {"type": "TRIP", "id": "made-up"},
                         {"type": "CHECKIN", "id": "c1"}, {"type": "FOOD", "id": "d1"},
                         {"type": "FOOD", "id": "d1", "citySlug": "da-lat"}, {"type": "PLACE", "id": "p7"}], ids)
    assert cards == [{"type": "TRIP", "id": "pt1"}, {"type": "CHECKIN", "id": "c1"},
                     {"type": "FOOD", "id": "d1", "citySlug": "da-lat"}]
    reply = clean_reply({"answer": "x", "actions": [{"action": "OPEN_TRIP", "label": "Mở", "tripId": "pt1"}],
                         "cards": [{"type": "TRIP", "id": "pt1"}]}, set(), {}, ids)
    assert reply["actions"] == [] and reply["cards"] == [{"type": "TRIP", "id": "pt1"}]


def test_destination_cards_need_an_explored_area_and_a_real_date_pair():
    names = {}
    ids = {"place": set(), "event": set(), "meetup": set(), "trip": set()}
    collect_names({"destinations": [{"id": "hg", "name": "Hà Giang", "kind": "destination"},
                                    {"id": "nb", "name": "Ninh Bình", "kind": "destination"}]}, names, ids)
    cards = clean_destinations([{"id": "hg", "note": "Mùa tam giác mạch", "startDate": "2026-11-07",
                                 "endDate": "2026-11-09"},
                                {"id": "hg"}, {"id": "made-up"},
                                {"id": "nb", "startDate": "tháng 11", "endDate": "2026-11-30"}], ids)
    assert cards == [{"id": "hg", "note": "Mùa tam giác mạch", "startDate": "2026-11-07", "endDate": "2026-11-09"},
                     {"id": "nb"}]
    assert step_for("get_destination_guide", {"destination_id": "hg"}, names) == ("destination_guide",
                                                                                  {"name": "Hà Giang"})
    assert {"explore_destinations", "get_destination_guide"} <= {t["function"]["name"] for t in tools_for("TRIP")}


def test_trip_scoped_community_tools_need_a_trip_only_in_the_direct_chat():
    for name in ("get_trip_checkins", "rank_trip_places"):
        trip_tool = next(t for t in tools_for("TRIP") if t["function"]["name"] == name)
        direct_tool = next(t for t in tools_for("ASSISTANT") if t["function"]["name"] == name)
        assert trip_tool["function"]["parameters"]["required"] == []
        assert direct_tool["function"]["parameters"]["required"] == ["trip_id"]
    assert step_for("search_meetups", {}, {}) == ("meetups", {})


@pytest.mark.asyncio
async def test_reasoning_model_with_tools_retries_with_reasoning_off(monkeypatch):
    import httpx
    from app import ai_log, providers
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
    monkeypatch.setattr(providers.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(providers, "_NO_REASONING_MODELS", set())
    monkeypatch.setattr(ai_log, "report", lambda **kw: _noop())
    for key, value in {"BASE_URL": "https://llm.test/v1", "API_KEY": "k", "MODEL": "gpt-test"}.items():
        monkeypatch.setenv(f"ASSISTANT_LLM_{key}", value)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]

    first = await chat_model.ChatModel().complete([{"role": "user", "content": "hi"}], tools)
    second = await chat_model.ChatModel().complete([{"role": "user", "content": "hi"}], tools)

    assert first["content"] == "ok" and second["content"] == "ok"
    assert [b.get("reasoning_effort") for b in sent] == [None, "none", "none"]


async def _noop():
    return None


@pytest.mark.asyncio
async def test_an_announcement_without_a_tool_call_is_not_posted_as_the_answer():
    java = FakeJava()
    model = ScriptedModel(
        {"content": "Mình xem thời tiết ngày 2 để gợi ý nhé."},
        {"content": "", "tool_calls": [tool_call("get_weather", {"day": 2})]},
        {"content": '{"answer": "Ngày 2 nắng, đi Bà Nà buổi sáng."}'},
    )
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})

    assert java.tools == [("get_weather", {"day": 2})]
    assert java.replies[0]["markdown"] == "Ngày 2 nắng, đi Bà Nà buổi sáng."
    assert model.calls[1]["messages"][-1]["role"] == "system"      # the nudge
    assert [s[0] for s in java.steps] == ["thought", "weather", "writing_answer"]


@pytest.mark.asyncio
async def test_plain_text_twice_is_accepted_as_the_answer():
    java = FakeJava()
    model = ScriptedModel({"content": "Xin chào!"}, {"content": "Xin chào bạn!"})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert java.replies[0]["markdown"] == "Xin chào bạn!" and len(model.calls) == 2


@pytest.mark.asyncio
async def test_actions_keep_only_ids_seen_and_memory_is_passed_on():
    class EventJava(FakeJava):
        async def tool(self, name, arguments):
            self.tools.append((name, arguments))
            return {"events": [{"id": "e1", "title": "Language exchange", "kind": "event"}]}

    java = EventJava()
    final = json.dumps({"answer": "Có buổi giao lưu nè", "memory": "  Thích cà phê yên tĩnh.  ", "actions": [
        {"action": "OPEN_EVENT", "label": "Xem sự kiện", "eventId": "e1"},
        {"action": "OPEN_EVENT", "label": "Bịa", "eventId": "e-invented"},
        {"action": "OPEN_TRIP", "label": "Mở chuyến", "tripId": "trip1"},
        {"action": "CREATE_TRIP", "label": "Tạo chuyến Huế", "city": "Huế", "startDate": "2026-10-16",
         "endDate": "bad"},
        {"action": "CREATE_TRIP", "label": "Thừa"}]}, ensure_ascii=False)
    model = ScriptedModel({"content": "", "tool_calls": [tool_call("search_events", {"day": 2})]}, {"content": final})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})

    reply = java.replies[0]
    assert reply["actions"] == [
        {"action": "OPEN_EVENT", "label": "Xem sự kiện", "eventId": "e1"},
        {"action": "OPEN_TRIP", "label": "Mở chuyến", "tripId": "trip1"},
        {"action": "CREATE_TRIP", "label": "Tạo chuyến Huế", "city": "Huế"},
    ]
    assert reply["memory"] == "Thích cà phê yên tĩnh."


@pytest.mark.asyncio
async def test_web_search_is_offered_only_when_allowed_and_runs_once(monkeypatch):
    from app.assistant import web_search

    searched = []

    async def fake_search(query):
        searched.append(query)
        return {"results": [{"title": "Blog", "url": "https://example.com", "snippet": "…"}]}

    monkeypatch.setattr(web_search, "configured", lambda: True)
    monkeypatch.setattr(web_search, "search", fake_search)
    assert "web_search" not in {t["function"]["name"] for t in tools_for("TRIP")}

    java = FakeJava()
    turn = {**TURN, "limits": {"maxToolCalls": 4, "maxOutputTokens": 800, "webSearch": True}}
    model = ScriptedModel(
        {"content": "", "tool_calls": [tool_call("web_search", {"query": "board game Huế"}, "a"),
                                       tool_call("web_search", {"query": "again"}, "b")]},
        {"content": '{"answer": "Theo [Blog](https://example.com)…"}'},
    )
    graph = build_graph(turn, java, model)
    await graph.ainvoke({"messages": initial_messages(turn), "tool_calls": 0})

    assert "web_search" in {t["function"]["name"] for t in model.calls[0]["tools"]}
    assert searched == ["board game Huế"]
    assert java.tools == []                                          # never sent to Java
    second = [m for m in model.calls[1]["messages"] if m.get("tool_call_id") == "b"][0]
    assert "not available" in second["content"]

    plain = FakeJava()
    model = ScriptedModel({"content": '{"answer": "ok"}'})
    await build_graph(TURN, plain, model).ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert "web_search" not in {t["function"]["name"] for t in model.calls[0]["tools"]}


def test_earlier_cards_are_named_in_history_and_may_be_cards_again():
    turn = {**TURN, "history": [{"role": "assistant", "text": "Hai quán nè",
                                 "places": [{"id": PLACE, "name": "Cộng Cà Phê"}]}]}
    messages = initial_messages(turn)
    assert "Cộng Cà Phê (" + PLACE + ")" in messages[1]["content"]
    reply = clean_reply({"answer": "x", "places": [{"placeId": PLACE}]}, {PLACE}, {})
    assert reply["places"] == [{"placeId": PLACE, "note": None}]


def test_memory_is_in_the_context():
    messages = initial_messages({**TURN, "memory": "Khởi hành từ Đà Nẵng."})
    assert "Khởi hành từ Đà Nẵng." in messages[-1]["content"]


@pytest.mark.asyncio
async def test_openai_web_search_keeps_only_links_the_search_returned(monkeypatch):
    from app.assistant import web_search

    monkeypatch.setenv("CHEAP_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CHEAP_LLM_API_KEY", "k")
    monkeypatch.setenv("CHEAP_LLM_MODEL", "m")
    monkeypatch.delenv("ASSISTANT_WEB_SEARCH_PROVIDER", raising=False)
    assert web_search.configured()
    body = {"output": [
        {"type": "web_search_call", "action": {"sources": [{"url": "https://real.vn/a", "title": "Real"}]}},
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps({"results": [
            {"title": "Real", "url": "https://real.vn/a", "snippet": "Board game tối thứ Sáu"},
            {"title": "Made up", "url": "https://fake.vn", "snippet": "x"}]})}]}]}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return body

    class Client:
        async def post(self, url, headers=None, json=None):
            assert url == "https://api.openai.com/v1/responses" and json["tools"][0]["type"] == "web_search"
            return Response()

    rows = await web_search._model_search(Client(), "board game Huế")
    assert rows == [{"title": "Real", "url": "https://real.vn/a", "snippet": "Board game tối thứ Sáu"}]


def test_long_plain_text_is_an_answer_but_an_announcement_is_not():
    from app.assistant.graph import looks_like_announcement
    assert looks_like_announcement("Mình xem thời tiết ngày 2 để gợi ý nhé.")
    assert looks_like_announcement("Let me check the weather for day 2 first.")
    assert looks_like_announcement("Xin chào!")
    long_answer = ("Vé thuyền Tràng An năm 2026 là **300.000đ/người lớn** (cao trên 1,3 m) và 150.000đ cho trẻ "
                   "em cao 1–1,3 m; trẻ dưới 1 m miễn phí. Giá có thể thay đổi, bạn kiểm tra lại trước khi đi.")
    assert not looks_like_announcement(long_answer)
    assert looks_like_announcement(long_answer + " Để mình kiểm tra thêm giờ mở cửa nhé")
    assert not looks_like_announcement("Phở Ngọc Béo mở 5:30–15:00.", acted=True)
    assert looks_like_announcement("Mình xem thêm giờ mở cửa nhé.", acted=True)


@pytest.mark.asyncio
async def test_a_long_plain_text_answer_is_posted_without_a_second_round():
    java = FakeJava()
    answer = ("Vé thuyền Tràng An năm 2026 là **300.000đ/người lớn** (cao trên 1,3 m) và 150.000đ cho trẻ "
              "em cao 1–1,3 m; trẻ dưới 1 m miễn phí. Giá có thể thay đổi, bạn kiểm tra lại trước khi đi.")
    model = ScriptedModel({"content": answer})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert len(model.calls) == 1
    assert java.replies[0]["markdown"] == answer
    assert [s[0] for s in java.steps] == ["writing_answer"]


@pytest.mark.asyncio
async def test_an_empty_json_answer_falls_back_to_the_text_round_before_it():
    java = FakeJava()
    model = ScriptedModel({"content": "Xin chào bạn, hôm nay mình giúp gì?"}, {"content": '{"answer": ""}'})
    graph = build_graph(TURN, java, model)
    await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert java.replies[0]["markdown"] == "Xin chào bạn, hôm nay mình giúp gì?"


@pytest.mark.asyncio
async def test_web_search_results_are_cached_per_query(monkeypatch, tmp_path):
    from app.assistant import web_search

    calls = []

    async def fake_openai(client, query):
        calls.append(query)
        return [{"title": "Real", "url": "https://real.vn/a", "snippet": "300.000đ"}]

    monkeypatch.setattr(web_search, "configured", lambda: True)
    monkeypatch.setattr(web_search, "_provider", lambda: "openai")
    monkeypatch.setattr(web_search, "_model_search", fake_openai)
    monkeypatch.setattr(web_search, "CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(web_search, "_cache", web_search.ResearchCache(str(tmp_path / "c.sqlite3")))
    monkeypatch.setattr(web_search, "_cache_broken", False)

    first = await web_search.search("giá vé Tràng An")
    second = await web_search.search("Giá vé tràng an")
    assert first["results"] == second["results"] and calls == ["giá vé Tràng An"]


def test_places_named_in_the_answer_become_cards():
    from app.assistant.graph import attach_mentioned_places, mentioned_places
    found = {"p1": "Phở Ngọc Béo Phố Cổ", "p2": "Pho", "p3": "Trúc Bạch Lake", "p4": "Hoàn Kiếm Lake"}
    answer = ("Nếu muốn ít đông, thử **hồ Trúc Bạch**. Còn phở thì ghé **Phở Ngọc Béo Phố Cổ**, "
              "phở ở đây ngon.")
    assert mentioned_places(answer, found) == ["p3", "p1"]          # "Pho" is too generic to match
    final = attach_mentioned_places({"answer": answer, "places": [{"placeId": "p1", "note": "x"}]}, found)
    assert final["places"] == [{"placeId": "p1", "note": "x"}, {"placeId": "p3"}]


@pytest.mark.asyncio
async def test_a_plain_text_answer_after_a_search_still_shows_the_place_card():
    class PlaceJava(FakeJava):
        async def tool(self, name, arguments):
            self.tools.append((name, arguments))
            return {"places": [{"id": "d79a", "name": "Phở Ngọc Béo Phố Cổ", "kind": "food"},
                               {"id": "31e9", "name": "Pho", "kind": "food"}]}

    java = PlaceJava()
    answer = ("Nếu muốn thử kiểu quán bình dân, bạn có thể ghé **Phở Ngọc Béo Phố Cổ**: giá khoảng "
              "1.000–100.000đ, mở hằng ngày 5:30–15:00. Giá và giờ có thể thay đổi.")
    model = ScriptedModel({"content": "", "tool_calls": [tool_call("search_places", {"query": "phở"})]},
                          {"content": answer})
    await build_graph(TURN, java, model).ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    assert java.replies[0]["markdown"] == answer
    assert [p["placeId"] for p in java.replies[0]["places"]] == ["d79a"]


def test_a_stray_bold_marker_is_dropped():
    from app.assistant.graph import balance_bold

    assert balance_bold("Bạn thích núi hay sông hơn?**") == "Bạn thích núi hay sông hơn?"
    assert balance_bold("**Mù Cang Chải** đẹp") == "**Mù Cang Chải** đẹp"


@pytest.mark.asyncio
async def test_tinyfish_search_returns_snippets(monkeypatch):
    from app.assistant import web_search

    monkeypatch.setenv("ASSISTANT_WEB_SEARCH_PROVIDER", "tinyfish")
    monkeypatch.setenv("ASSISTANT_WEB_SEARCH_API_KEY", "tf-key")
    assert web_search.configured()

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"results": [{"position": 1, "title": "Lễ hội", "url": "https://hue.vn/a",
                                 "snippet": "<b>Festival</b> Huế 2026"},
                                {"title": "No link"}]}

        def raise_for_status(self):
            pass

    class Client:
        async def get(self, url, headers, params):
            assert url == "https://api.search.tinyfish.ai"
            assert headers == {"X-API-Key": "tf-key"} and params == {"query": "festival Huế"}
            return Response()

    rows = await web_search._tinyfish(Client(), "festival Huế")
    assert rows == [{"title": "Lễ hội", "url": "https://hue.vn/a", "snippet": "Festival Huế 2026"}]


@pytest.mark.asyncio
async def test_a_failing_search_api_falls_back_to_openai(monkeypatch):
    from app.assistant import web_search

    monkeypatch.setenv("ASSISTANT_WEB_SEARCH_PROVIDER", "tinyfish")
    monkeypatch.setenv("ASSISTANT_WEB_SEARCH_API_KEY", "tf-key")
    monkeypatch.setenv("CHEAP_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CHEAP_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("CHEAP_LLM_MODEL", "gpt-x")
    monkeypatch.setattr(web_search, "CACHE_TTL_SECONDS", 0)
    # The tinyfish key is not an OpenAI key: the fallback runs on CHEAP_LLM_*.
    endpoint = web_search._model_endpoint()
    assert (endpoint.base_url, endpoint.api_key, endpoint.model) == ("https://api.openai.com/v1", "sk-test", "gpt-x")

    async def rate_limited(client, query):
        raise RuntimeError("HTTP 429")

    async def fake_openai(client, query):
        return [{"title": "Real", "url": "https://real.vn/a", "snippet": "300.000đ"}]

    monkeypatch.setitem(web_search._SEARCHERS, "tinyfish", rate_limited)
    monkeypatch.setattr(web_search, "_model_search", fake_openai)
    reply = await web_search.search("giá vé Tràng An")
    assert reply["results"][0]["url"] == "https://real.vn/a"


def test_prose_followed_by_json_keeps_the_prose_as_the_answer():
    content = ('Miền Trung tháng 10 thì **Huế** hợp nhất {nhé}.\n\nBạn muốn đi biển hay văn hóa?\n'
               '{"destinations":[{"id":"d1","note":"Cố đô"}],"suggestions":["Gợi ý lịch trình Huế 2 ngày"]}')
    final = parse_final(content)
    assert final["answer"] == "Miền Trung tháng 10 thì **Huế** hợp nhất {nhé}.\n\nBạn muốn đi biển hay văn hóa?"
    assert final["destinations"] == [{"id": "d1", "note": "Cố đô"}]
    assert final["suggestions"] == ["Gợi ý lịch trình Huế 2 ngày"]
    fenced = parse_final('Huế nhé!\n```json\n{"suggestions": ["Ăn gì?"]}\n```')
    assert fenced == {"suggestions": ["Ăn gì?"], "answer": "Huế nhé!"}


def test_thin_catalogue_results_point_to_the_web():
    def result(names, words=None, relaxed=None):
        out = {"searched": {"nameWords": words} if words else {}, "places": [{"id": n, "name": n} for n in names]}
        if relaxed:
            out["relaxed"] = relaxed
        return out

    # The asked words match only one row: the rest are same-kind padding.
    padded = result(["Phở Chay Hà Nội", "Chayfood", "Bonbagu Vegan"], "phở chay", ["added places of the same kind"])
    assert graph_module.flag_thin_catalogue(padded, True)["coverage"] == "thin"
    # A widened radius is thin however many rows came back.
    far = result(["A", "B", "C"], relaxed=["widened the radius to 30 km"])
    assert graph_module.flag_thin_catalogue(far, True)["coverage"] == "thin"
    assert graph_module.flag_thin_catalogue(result(["A"]), True)["coverage"] == "thin"
    # Enough real matches, or no web search left: left as it came.
    full = result(["Phở Thìn", "Phở Bát Đàn", "Phở 10 Lý Quốc Sư"], "phở")
    assert "coverage" not in graph_module.flag_thin_catalogue(full, True)
    assert "coverage" not in graph_module.flag_thin_catalogue(result(["A"]), False)


def test_restaurants_far_from_the_stop_do_not_answer_a_beach_market_question():
    # Prod 2026-10-25: "chợ buổi sáng ở bãi biển… có gì ăn" near a café on day 1.
    rows = [{"id": n, "name": n, "types": ["RESTAURANT", "SEAFOOD"], "distanceKm": d}
            for n, d in (("Garden House", 3.7), ("Quán Nhậu 6 Vân", 3.7), ("Nhà Bè Bảo An", 2.9))]
    result = {"near": "Tiệm Cà Phê Nhà Ven Biển", "searched": {"types": ["SEAFOOD", "STREET_FOOD", "MARKET"]},
              "places": rows}
    flagged = graph_module.flag_thin_catalogue(result, True, {"activity_id": "a1"})
    assert flagged["coverage"] == "thin"
    assert "nothing of type MARKET" in flagged["note"] and "STREET_FOOD" not in flagged["note"]
    assert "within 1.5 km" in flagged["note"]
    # Sibling food types asked as alternatives are not a missing part.
    cafes = {"searched": {"types": ["CAFE", "TEA"]},
             "places": [{"id": n, "name": n, "types": ["CAFE"]} for n in "ABC"]}
    assert "coverage" not in graph_module.flag_thin_catalogue(cafes, True)


@pytest.mark.asyncio
async def test_an_answer_over_a_thin_search_is_sent_back_once_to_search_the_web(monkeypatch):
    from app.assistant import web_search

    async def fake_search(query):
        return {"results": [{"title": "Chợ cá Dốc Lết", "url": "https://x.vn/a", "snippet": "Chợ cá 5h sáng"}]}

    monkeypatch.setattr(web_search, "configured", lambda: True)
    monkeypatch.setattr(web_search, "search", fake_search)
    java = FakeJava()   # one row only: thin
    turn = {**TURN, "limits": {"maxToolCalls": 4, "webSearch": True}}
    weak = '{"answer": "Mình chưa có thông tin, bạn hỏi người dân nhé."}'
    model = ScriptedModel({"content": "", "tool_calls": [tool_call("search_places", {"types": ["MARKET"]})]},
                          {"content": weak},
                          {"content": "", "tool_calls": [tool_call("web_search", {"query": "chợ cá Dốc Lết"}, "w1")]},
                          {"content": '{"answer": "Có, **chợ cá** họp lúc 5h sáng."}'})
    graph = build_graph(turn, java, model)
    await graph.ainvoke({"messages": initial_messages(turn), "tool_calls": 0})
    pushes = [m for m in model.calls[2]["messages"] if m.get("content") == graph_module.WEB_PUSH]
    assert len(pushes) == 1 and len(model.calls) == 4
    assert java.replies[0]["markdown"] == "Có, **chợ cá** họp lúc 5h sáng."


def test_read_url_keeps_the_shop_names_of_a_list_article():
    from app.assistant.read_url import html_to_text
    title, text = html_to_text(
        "<html><head><title>Top 3 quán bánh tráng cuốn</title><script>var x=1</script></head><body>"
        "<nav>Trang chủ</nav><h2>1. Bánh tráng cuốn Hoàng Bèo</h2><p>Số 9 ngõ 3 Quang Trung</p>"
        "<ul><li>Giá: 50k</li></ul><footer>© báo</footer></body></html>")
    assert title == "Top 3 quán bánh tráng cuốn"
    assert "## 1. Bánh tráng cuốn Hoàng Bèo" in text and "Số 9 ngõ 3 Quang Trung" in text and "- Giá: 50k" in text
    assert "var x" not in text and "Trang chủ" not in text and "© báo" not in text


@pytest.mark.asyncio
async def test_read_url_refuses_private_addresses_and_odd_links():
    from app.assistant import read_url
    assert "error" in await read_url.read("http://127.0.0.1:8080/admin")
    assert "error" in await read_url.read("http://10.0.0.5/")
    assert "error" in await read_url.read("file:///etc/passwd")
    assert "error" in await read_url.read("not a link")


def test_reading_pages_comes_with_web_search():
    names = {t["function"]["name"] for t in tools_for("TRIP", web_search=True)}
    assert {"web_search", "read_url"} <= names
    assert "read_url" not in {t["function"]["name"] for t in tools_for("TRIP")}
    assert step_for("read_url", {"url": "https://www.toplist.vn/top-11"}, {}) == ("read_page", {"name": "toplist.vn"})


class ReportingModel(ScriptedModel):
    """Logs each call like the real chat model, so Java's credit answer reaches the turn."""

    async def complete(self, messages, tools=None, **kwargs):
        reply = await super().complete(messages, tools, **kwargs)
        await ai_log.report(operation="ASSISTANT_TURN", provider="p", model="m", ok=True, request={})
        return reply


@pytest.mark.asyncio
async def test_running_out_of_credit_stops_with_what_was_written():
    java = FakeJava()

    async def sink(entry):
        return {"creditBalance": -12, "creditExhausted": True}

    binding = ai_log.bind(sink, feature="ASSISTANT")
    try:
        model = ReportingModel(
            {"content": "Mình đang tìm quán bún chả cá gần khách sạn.",
             "tool_calls": [tool_call("search_places", {"query": "bún chả cá"})]},
            {"content": FINAL})
        graph = build_graph(TURN, java, model)
        await graph.ainvoke({"messages": initial_messages(TURN), "tool_calls": 0})
    finally:
        ai_log.unbind(binding)

    assert len(model.calls) == 1                     # no model call after the credit ran out
    reply = java.replies[0]
    assert reply["stopReason"] == "CREDIT_EXHAUSTED"
    assert reply["markdown"] == "Mình đang tìm quán bún chả cá gần khách sạn."
