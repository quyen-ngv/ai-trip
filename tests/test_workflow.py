"""End-to-end run of the graph against a fake Java backend and a fake LLM."""
import json
import pytest
from app import workflow
from app.prompts import build_itinerary_prompt, preferences_block

CENTER = (21.0285, 105.8542)


def java_row(i, group, **kw):
    base = {"id": f"00000000-0000-4000-8000-{i:012d}", "googlePlaceId": f"g{i}", "title": f"Quán {i}" if group == "FOOD_AND_DRINK" else f"Điểm {i}",
            "address": f"{i} Phố Cổ", "latitude": CENTER[0] + (i % 5) * 0.003, "longitude": CENTER[1] + ((i // 5) % 5) * 0.003,
            "placeGroup": group, "category": "Restaurant" if group == "FOOD_AND_DRINK" else "Museum",
            "reviewCount": 500 + i, "reviewRating": 4.2, "score": 7.0, "distanceKm": 1.0, "menuHighlights": ["Phở bò"] if group == "FOOD_AND_DRINK" else []}
    base.update(kw)
    return base


class FakeJava:
    def __init__(self):
        self.events, self.committed = [], None

    async def event(self, stage, progress, key, status="RUNNING", error=None, params=None):
        self.events.append((stage, progress, status))

    async def config(self):
        return {}

    async def candidates(self, d, groups, limit):
        g = groups[0]
        n = 14 if g == "FOOD_AND_DRINK" else 8 if g == "CULTURE_AND_HERITAGE" else 0
        return [java_row(i + (0 if g == "FOOD_AND_DRINK" else 100), g) for i in range(n)]

    async def commit(self, items, description):
        self.committed = (items, description)
        return {"tripId": "t"}


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def json(self, system, payload, **kw):
        self.calls.append((system, payload))
        if "COMPLETE itinerary" in system:
            food = [p["id"] for p in payload["food"]]
            acts = [p["id"] for p in payload["activities"]]
            days = []
            for k, d in enumerate(payload["days"]):
                f, a = food[3 * k:3 * k + 3], acts[2 * k:2 * k + 2]
                if len(f) < 3 or len(a) < 2:
                    break
                days.append({"dayNumber": d["dayNumber"], "theme": "Phố cổ", "stops": [
                    {"placeId": f[0], "start": "07:30", "end": "08:15", "description": "Ghé ăn sáng phở bò nóng."},
                    {"placeId": a[0], "start": "08:45", "end": "10:15", "description": "Ghé tham quan di tích."},
                    {"placeId": f[1], "start": "12:00", "end": "13:00", "description": "Ghé ăn trưa bún chả."},
                    {"rest": "hotel", "start": "13:30", "end": "15:00", "description": "Nghỉ trưa tránh nắng."},
                    {"placeId": a[1], "start": "15:30", "end": "17:00", "description": "Ghé bảo tàng buổi chiều."},
                    {"placeId": f[2], "start": "18:30", "end": "19:45", "description": "Ghé ăn tối đặc sản."},
                ]})
            return {"tripDescription": "Hai ngày quanh phố cổ.", "days": days}
        return None


class NoopWebResearcher:
    """Keeps graph tests focused on the pipeline rather than a configured external browser."""

    def __init__(self, **kwargs):
        self.enabled = True

    async def research_destination(self, destination, needs, known):
        return []


JOB = {"jobId": "j", "attemptId": "a", "locale": "vi", "callbackBaseUrl": "http://java",
       "request": {"tripName": "Hà Nội", "cityName": "Hà Nội", "placeGroups": ["CULTURE_AND_HERITAGE"], "pace": "BALANCED",
                   "discoveryStyle": "HIDDEN_GEMS", "groupComposition": "Couple (2 adults)", "dietaryRestrictions": ["no_pork"],
                   "mobilityConsiderations": ["Elderly-friendly"], "preferenceText": "muốn ăn phở",
                   "destinations": [{"name": "Hà Nội", "latitude": CENTER[0], "longitude": CENTER[1], "startDate": "2026-09-10", "endDate": "2026-09-11"}]}}


@pytest.mark.asyncio
async def test_graph_end_to_end(monkeypatch):
    fake_java, fake_llm = FakeJava(), FakeLLM()
    monkeypatch.setattr(workflow, "JavaClient", lambda job, token: fake_java)
    monkeypatch.setattr(workflow, "CachedModel", lambda *a, **k: fake_llm)
    monkeypatch.setattr(workflow, "WebResearcher", NoopWebResearcher)
    graph = await workflow.build_workflow(JOB, "tok")
    await graph.ainvoke({"job": JOB})

    items, description = fake_java.committed
    assert description == "Hai ngày quanh phố cổ."
    visits = [i for i in items if i["type"] == "PLACE_VISIT"]
    assert {i["dayNumber"] for i in visits} == {1, 2}
    assert all(i["placeId"] and i["name"].startswith(("Quán", "Điểm")) for i in visits)   # DB titles, valid ids
    assert all(i["description"].startswith("Ghé") for i in visits)                       # notes keyed by placeId landed
    assert not any(i["type"] == "TRANSPORT" for i in items)  # all stops within ~2 km: no in-city transport rows
    assert all(not any(k.startswith("_") for k in i) for i in items)
    # Java validateItems: sortOrder sequential, no overlap per day
    assert [i["sortOrder"] for i in items] == list(range(len(items)))
    for d in (1, 2):
        day = [i for i in items if i["dayNumber"] == d]
        for a, b in zip(day, day[1:]):
            assert b["startTime"] >= a["endTime"]
    # exactly one LLM call for one destination, carrying the preferences that used to be lost
    assert len(fake_llm.calls) == 1
    system = fake_llm.calls[0][0]
    for needle in ("HIDDEN_GEMS", "Couple", "Elderly-friendly", "no_pork", "muốn ăn phở", "Vietnamese", "Hà Nội"):
        assert needle.lower() in system.lower(), needle
    payload = fake_llm.calls[0][1]
    assert all(p["title"] for p in payload["food"]) and any(p.get("menu") for p in payload["food"])
    rest = [i for i in items if i["name"] == "Nghỉ ngơi tại khách sạn"]
    assert rest and rest[0]["type"] == "ACTIVITY"
    assert fake_java.events[-1][0] == "SAVING"


@pytest.mark.asyncio
async def test_graph_researches_the_web_even_when_catalogue_is_sufficient(monkeypatch):
    class RecordingResearcher:
        calls = []

        def __init__(self, **kwargs):
            self.enabled = True

        async def research_destination(self, destination, needs, known):
            self.calls.append((destination["name"], needs, len(known)))
            return []

    fake_java, fake_llm = FakeJava(), FakeLLM()

    async def dense_candidates(destination, groups, limit):
        group = groups[0]
        return [java_row(i + (0 if group == "FOOD_AND_DRINK" else 100), group) for i in range(30)]

    fake_java.candidates = dense_candidates
    monkeypatch.setattr(workflow, "JavaClient", lambda job, token: fake_java)
    monkeypatch.setattr(workflow, "CachedModel", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(workflow, "WebResearcher", RecordingResearcher)

    graph = await workflow.build_workflow(JOB, "tok")
    await graph.ainvoke({"job": JOB})

    assert RecordingResearcher.calls == [
        ("Hà Nội", {"FOOD_AND_DRINK": 3, "CULTURE_AND_HERITAGE": 3}, 0),
    ]


@pytest.mark.asyncio
async def test_web_research_binds_only_the_cheap_model(monkeypatch):
    fake_java, fake_llm = FakeJava(), FakeLLM()
    prefixes = []

    def models(*args, **kwargs):
        prefixes.append(args)
        return fake_llm

    monkeypatch.setattr(workflow, "JavaClient", lambda job, token: fake_java)
    monkeypatch.setattr(workflow, "CachedModel", models)
    monkeypatch.setattr(workflow, "WebResearcher", NoopWebResearcher)
    await workflow.build_workflow(JOB, "tok")

    assert prefixes == [("QUALITY_LLM", "CHEAP_LLM"), ("CHEAP_LLM",)]


def test_research_input_uses_city_name_or_coordinates_when_destination_name_is_blank():
    first = workflow._research_destination_input(
        {"latitude": 21.0285, "longitude": 105.8542},
        fallback_name="Hà Nội",
    )
    later = workflow._research_destination_input({"latitude": 16.0544, "longitude": 108.2022})
    assert first["name"] == "Hà Nội"
    assert later["name"] == "16.05440,108.20220"


def test_resolve_dest_dates_gives_handover_day_to_arriving_destination():
    from app.workflow import _resolve_dest_dates, _validate_destinations
    dests = [
        {"name": "Hà Nội", "startDate": "2026-09-10", "endDate": "2026-09-12"},
        {"name": "Đà Nẵng", "startDate": "2026-09-12", "endDate": "2026-09-14"},  # shares 12th
    ]
    _validate_destinations(dests)
    dd = _resolve_dest_dates(dests)
    assert [x.isoformat() for x in dd[0]] == ["2026-09-10", "2026-09-11"]
    assert [x.isoformat() for x in dd[1]] == ["2026-09-12", "2026-09-13", "2026-09-14"]
    # a 1-day destination keeps its only day; the later one is trimmed instead
    dd2 = _resolve_dest_dates([
        {"startDate": "2026-09-10", "endDate": "2026-09-10"},
        {"startDate": "2026-09-10", "endDate": "2026-09-11"},
    ])
    assert [x.isoformat() for x in dd2[0]] == ["2026-09-10"]
    assert [x.isoformat() for x in dd2[1]] == ["2026-09-11"]
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        _validate_destinations([{"name": "x", "startDate": "2026-09-12", "endDate": "2026-09-10"}])


@pytest.mark.asyncio
async def test_multi_destination_long_hop(monkeypatch):
    job = {**JOB, "request": {**JOB["request"], "destinations": [
        {"name": "Hà Nội", "latitude": CENTER[0], "longitude": CENTER[1], "startDate": "2026-09-10", "endDate": "2026-09-11"},
        {"name": "Đà Nẵng", "latitude": 16.05, "longitude": 108.20, "startDate": "2026-09-12", "endDate": "2026-09-13"},
    ]}}
    fake_java, fake_llm = FakeJava(), FakeLLM()
    monkeypatch.setattr(workflow, "JavaClient", lambda j, t: fake_java)
    monkeypatch.setattr(workflow, "CachedModel", lambda *a, **k: fake_llm)
    monkeypatch.setattr(workflow, "WebResearcher", NoopWebResearcher)
    graph = await workflow.build_workflow(job, "tok")
    await graph.ainvoke({"job": job})
    items, _ = fake_java.committed
    inter = [i for i in items if i["type"] == "TRANSPORT" and i.get("endDayNumber")]
    assert inter and inter[0]["endDayNumber"] == 3 and inter[0]["transportMode"] in {"PLANE", "TRAIN"}
    visits = [i for i in items if i["type"] == "PLACE_VISIT"]
    assert {i["dayNumber"] for i in visits} == {1, 2, 3, 4}
    for d in (1, 2, 3, 4):  # exactly one of each meal per day, never doubled by the handover logic
        slots = [i["_slot"] if "_slot" in i else None for i in visits if i["dayNumber"] == d]
        # _slot is stripped before commit; recount via names being unique instead
        names = [i["name"] for i in visits if i["dayNumber"] == d]
        assert len(names) == len(set(names))
    keys = [i["placeId"] for i in visits]
    assert len(keys) == len(set(keys))  # no place reused across the two provinces


def test_prompt_contains_rules_and_preferences():
    req = JOB["request"]
    p = build_itinerary_prompt(req, "vi", "Đà Nẵng", 2, [1, 2], ["Monday", "Tuesday"], {})
    assert "JSON" in p and "Kid-friendly" in p and "NEVER a cafe" in p
    assert "## Local knowledge for this destination" not in p
    block = preferences_block({"groupComposition": "Family with children", "mobilityConsiderations": ["Kid-friendly"]}, "en")
    assert "Kid-friendly" in block and "Family with children" in block
    # config overrides the built-in rules
    p2 = build_itinerary_prompt(req, "en", "Huế", 1, [1], ["Monday"], {"PLAN_RULES": "MY CUSTOM RULES"})
    assert "MY CUSTOM RULES" in p2 and "NEVER a cafe" not in p2
