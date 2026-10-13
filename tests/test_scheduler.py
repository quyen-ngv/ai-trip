from datetime import date
from app.candidates import normalize
from app.scheduler import MEAL_OF, schedule_destination
from app.validators import check, finalize, strip_internal

CENTER = {"latitude": 16.0544, "longitude": 108.2022}  # Da Nang


def row(i, group, **kw):
    base = {
        "id": f"00000000-0000-4000-8000-{i:012d}", "googlePlaceId": f"g{i}", "title": f"Place {i}",
        "latitude": CENTER["latitude"] + (i % 7) * 0.004, "longitude": CENTER["longitude"] + (i // 7) * 0.004,
        "placeGroup": group, "reviewRating": 4.0 + (i % 5) / 10, "reviewCount": 100 + i * 10, "category": "",
    }
    base.update(kw)
    return normalize(base)


def inventory():
    food = [row(i, "FOOD_AND_DRINK", category="Restaurant") for i in range(1, 13)]
    food[0]["meals"] = ["breakfast"]
    acts = [row(i, "CULTURE_AND_HERITAGE", category="Museum", openHours={"Monday": ["8 AM–5 PM"]}) for i in range(20, 28)]
    acts += [row(i, "NATURE_AND_OUTDOORS", category="Park") for i in range(30, 36)]
    return food + acts


def test_schedule_fills_meals_and_orders_times():
    rows = inventory()
    used = set()
    items, days, notes = schedule_destination(rows, [1, 2], [date(2026, 9, 7), date(2026, 9, 8)], "BALANCED", CENTER, "vi", used)
    visits = [i for i in items if i["type"] != "TRANSPORT"]
    for d in (1, 2):
        slots = [i["_slot"] for i in visits if i["dayNumber"] == d]
        for meal in MEAL_OF:
            assert meal in slots, (d, slots)
        assert 2 <= len([s for s in slots if s not in MEAL_OF]) <= 4
    # times are strictly increasing within a day and nobody is visited twice
    for d in (1, 2):
        day = [i for i in items if i["dayNumber"] == d]
        assert [i["startTime"] for i in day] == sorted(i["startTime"] for i in day)
        for a, b in zip(day, day[1:]):
            assert b["startTime"] >= a["endTime"]
    keys = [i["placeId"] for i in visits]
    assert len(keys) == len(set(keys))
    # activity names are the DB titles, never invented
    assert all(i["name"].startswith("Place ") for i in visits)
    # stops here are all within ~2 km: no in-city TRANSPORT rows
    assert not any(i["type"] == "TRANSPORT" for i in items)
    assert check(items, "BALANCED", 2) == []


def test_schedule_respects_opening_hours():
    rows = inventory()
    used = set()
    items, days, notes = schedule_destination(rows, [1], [date(2026, 9, 7)], "EAGER", CENTER, "en", used)  # Monday
    visits = [i for i in items if i["type"] != "TRANSPORT"]
    assert [i["_slot"] for i in visits if i["_slot"] in MEAL_OF] == ["BREAKFAST", "LUNCH", "DINNER"]
    museums = [i for i in visits if i["category"] == "attraction"]
    for m in museums:  # museums are open 8-17 on Monday: never scheduled at night
        assert m["endTime"] <= "17:00:00"
    assert visits[0]["startTime"] >= "07:00:00" and visits[-1]["endTime"] <= "23:00:00"


def test_cafe_never_fills_breakfast():
    cafe = normalize({"id": "00000000-0000-4000-8000-000000000777", "googlePlaceId": "gc", "title": "Cafe X",
                      "latitude": CENTER["latitude"], "longitude": CENTER["longitude"],
                      "placeGroup": "FOOD_AND_DRINK", "category": "Coffee shop", "reviewRating": 4.9, "reviewCount": 90000})
    real = normalize({"id": "00000000-0000-4000-8000-000000000778", "googlePlaceId": "gr", "title": "Bún bò O Hoa",
                      "latitude": CENTER["latitude"], "longitude": CENTER["longitude"],
                      "placeGroup": "FOOD_AND_DRINK", "category": "Restaurant", "reviewRating": 3.9, "reviewCount": 50})
    acts = [normalize({"id": f"00000000-0000-4000-8000-00000000090{i}", "googlePlaceId": f"ga{i}", "title": f"Điểm {i}",
                       "latitude": CENTER["latitude"], "longitude": CENTER["longitude"],
                       "placeGroup": "CULTURE_AND_HERITAGE", "reviewRating": 4.4, "reviewCount": 500}) for i in range(3)]
    items, _, _ = schedule_destination([cafe, real] + acts, [1], [date(2026, 9, 8)], "BALANCED", CENTER, "vi", set())
    breakfast = [i for i in items if i.get("_slot") == "BREAKFAST"]
    assert breakfast and breakfast[0]["name"] == "Bún bò O Hoa"  # the cafe lost despite far better rating


def test_google_only_rows_become_activity_items_with_google_name():
    rows = inventory()[:6]
    extra = normalize({"id": None, "googlePlaceId": "gX", "title": "Bà Nà Hills", "latitude": 15.9977, "longitude": 107.9880,
                       "placeGroup": "ATTRACTIONS", "reviewRating": 4.5, "reviewCount": 40000}, source="google")
    items, _, _ = schedule_destination(rows + [extra], [1], [date(2026, 9, 9)], "RELAXED", CENTER, "en", set())
    acts = [i for i in items if i["type"] == "ACTIVITY"]
    assert acts and acts[0]["name"] == "Bà Nà Hills" and "placeId" not in acts[0] and acts[0]["latitude"]


def test_scheduler_keeps_all_social_options_in_one_choice_window():
    options = [
        row(101, "FOOD_AND_DRINK", title="Lunch A", category="Restaurant", candidateRef="social-1",
            socialJobId="social-job", socialSequence=1, optionGroupId="lunch", optionIndex=1,
            relation="OPTION", resolutionStatus="RESOLVED"),
        row(102, "FOOD_AND_DRINK", title="Lunch B", category="Restaurant", candidateRef="social-2",
            socialJobId="social-job", socialSequence=2, optionGroupId="lunch", optionIndex=2,
            relation="OPTION", resolutionStatus="RESOLVED"),
    ]
    options = [{**option, "source": "social"} for option in options]
    items, _, _ = schedule_destination(options, [1], [date(2026, 9, 8)], "BALANCED", CENTER, "en", set())
    visits = [item for item in items if item["type"] != "TRANSPORT"
              and item.get("sourceSocialCandidateRef") in {"social-1", "social-2"}]

    assert [item["sourceSocialCandidateRef"] for item in visits] == ["social-1", "social-2"]
    assert visits[0]["startTime"] == visits[1]["startTime"]
    assert visits[0]["endTime"] == visits[1]["endTime"]


def test_scheduler_preserves_social_video_order_across_slots():
    first = row(201, "ATTRACTIONS", title="Video point one", candidateRef="social-1",
                socialJobId="social-job", socialSequence=1, resolutionStatus="UNRESOLVED")
    second = row(202, "ATTRACTIONS", title="Video point two", candidateRef="social-2",
                 socialJobId="social-job", socialSequence=2, resolutionStatus="UNRESOLVED")
    rows = [{**first, "id": None, "source": "social"}, {**second, "id": None, "source": "social"}]
    items, _, _ = schedule_destination(rows, [1], [date(2026, 9, 8)], "BALANCED", CENTER, "en", set())
    visits = [item for item in items if item.get("sourceSocialCandidateRef")]

    assert [item["sourceSocialCandidateRef"] for item in visits] == ["social-1", "social-2"]


def test_finalize_matches_java_validate_items():
    items = [
        {"type": "PLACE_VISIT", "placeId": "a", "name": "A", "dayNumber": 1, "sortOrder": 5, "startTime": "09:00:00", "endTime": "10:00:00", "_slot": "MORNING"},
        {"type": "PLACE_VISIT", "placeId": "b", "name": "B", "dayNumber": 1, "sortOrder": 1, "startTime": "09:30:00", "endTime": "10:30:00"},
        {"type": "PLACE_VISIT", "placeId": None, "name": "C", "dayNumber": 1, "sortOrder": 2, "startTime": "11:00:00", "endTime": "12:00:00"},
        {"type": "PLACE_VISIT", "placeId": "d", "name": "D", "dayNumber": 9, "sortOrder": 3, "startTime": "11:00:00", "endTime": "12:00:00"},
        {"type": "TRANSPORT", "name": "T", "dayNumber": 1, "endDayNumber": 2, "sortOrder": 4, "startTime": "20:00:00", "endTime": "08:00:00"},
    ]
    out = strip_internal(finalize(items, 2))
    assert [i["name"] for i in out] == ["A", "B", "T"]
    assert out[1]["startTime"] == "10:00:00" and out[1]["endTime"] == "11:00:00"  # overlap pushed forward
    assert [i["sortOrder"] for i in out] == [0, 1, 2]
    assert "_slot" not in out[0] and "placeId" in out[0]
