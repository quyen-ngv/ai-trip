from app.candidates import (FOOD, activity_category, compact_for_plan, meal_hint, merge, normalize, quality,
                            research_needs, shortage)

JAVA_ROW = {  # exact shape of AiTripGenerationService.candidates()
    "id": "9c1c0b2e-0000-4000-8000-000000000001", "googlePlaceId": "ChIJabc", "title": "Phở Thìn Bờ Hồ",
    "address": "61 Đinh Tiên Hoàng", "latitude": 21.0301, "longitude": 105.8525, "placeGroup": "FOOD_AND_DRINK",
    "category": "Pho restaurant", "reviewCount": 2345, "reviewRating": 4.3, "score": 7.9, "distanceKm": 0.4,
    "visitDurationMinutes": None, "description": "  Famous   pho since 1955 " + "x" * 400,
    "attributes": {"servesBreakfast": {"value": True}, "wheelchair": False, "priceLevel": "$"},
    "menuHighlights": ["Phở tái", "Phở chín"], "openHours": {"Monday": ["6 AM–9 PM"]},
}


def test_normalize_reads_java_field_names():
    r = normalize(JAVA_ROW)
    assert r["title"] == "Phở Thìn Bờ Hồ"           # the name that ends up on the activity
    assert r["reviewRating"] == 4.3 and r["reviewCount"] == 2345 and r["score"] == 7.9
    assert r["menuHighlights"] == ["Phở tái", "Phở chín"]
    assert len(r["description"]) <= 300 and "  " not in r["description"]
    assert "servesBreakfast" in r["attributes"] and "wheelchair" not in r["attributes"]
    assert r["openHours"] == {"Monday": ["6 AM–9 PM"]}
    c = compact_for_plan(r, ["Monday"])
    assert c["title"] == "Phở Thìn Bờ Hồ" and c["rating"] == 4.3 and c["reviews"] == 2345 and c["menu"]
    assert c["hours"].startswith("Mon ")


def test_quality_prefers_rated_and_reviewed():
    good = normalize(JAVA_ROW)
    weak = normalize({**JAVA_ROW, "id": "x", "reviewRating": 3.2, "reviewCount": 4, "score": 2.0})
    assert quality(good) > quality(weak)


def test_merge_dedupes_by_id_and_google_id():
    a = normalize(JAVA_ROW)
    b = normalize({**JAVA_ROW, "id": None, "googlePlaceId": "ChIJabc"})
    c = normalize({**JAVA_ROW, "id": "other", "googlePlaceId": "ChIJzzz"})
    assert [r["id"] for r in merge([a], [b, c])] == [a["id"], "other"]


def test_shortage_asks_for_food_and_selected_groups_only():
    rows = [normalize({**JAVA_ROW, "id": str(i)}) for i in range(4)]  # 4 food, nothing else
    need = shortage(rows, "BALANCED", 2, ["CULTURE_AND_HERITAGE"])
    assert need[FOOD] == 12 - 4
    assert need["CULTURE_AND_HERITAGE"] == 15  # ceil(3*2*2.5 / 1)
    assert "NATURE_AND_OUTDOORS" not in need


def test_research_needs_enriches_even_a_sufficient_catalogue():
    rows = [
        normalize({**JAVA_ROW, "id": f"food-{i}"}) for i in range(20)
    ] + [
        normalize({**JAVA_ROW, "id": f"culture-{i}", "googlePlaceId": f"culture-google-{i}",
                   "placeGroup": "CULTURE_AND_HERITAGE"}) for i in range(20)
    ]
    needs = research_needs(rows, "BALANCED", 1, ["CULTURE_AND_HERITAGE"], minimum_per_group=3)
    assert needs == {FOOD: 3, "CULTURE_AND_HERITAGE": 3}
    assert research_needs(rows, "RELAXED", 1, ["CULTURE_AND_HERITAGE"], minimum_per_group=3)[FOOD] == 2
    assert research_needs(rows, "EAGER", 1, ["CULTURE_AND_HERITAGE"], minimum_per_group=3)[FOOD] == 4


def test_category_and_meal_hints():
    assert activity_category({"placeGroup": "CULTURE_AND_HERITAGE", "category": "Buddhist temple", "title": "Chùa Trấn Quốc"}) == "spiritual"
    assert activity_category({"placeGroup": "NATURE_AND_OUTDOORS", "category": "Beach", "title": "Bãi biển Mỹ Khê"}) == "beach"
    assert activity_category({"placeGroup": FOOD, "category": "Cafe", "title": "x"}) == "restaurant"
    assert meal_hint({"category": "Coffee shop", "title": "Cộng Cà Phê"}) == ["snack"]  # cafe is never breakfast
    assert meal_hint({"category": "Restaurant", "title": "Bún chả Hương Liên"}) == ["breakfast", "lunch"]
    assert meal_hint({"category": "Bar", "title": "Sky bar"}) == ["dinner"]
