from app.workflow import rank, solve_slots, destination_days

def test_rank_uses_menu_and_preferences():
    pho={"title":"Quán A","menuHighlights":["Phở bò"],"reviewRating":4.0,"distanceKm":2}
    goat={"title":"Quán B","menuHighlights":["Thịt dê"],"reviewRating":4.8,"distanceKm":1}
    assert rank(pho,"muốn ăn phở")>rank(goat,"muốn ăn phở")

def test_solver_caps_places_to_day_capacity():
    places=[{"id":str(i),"title":f"P{i}","score":10-i,"distanceKm":i} for i in range(20)]
    assert len(solve_slots(places,2,""))==6

def test_destination_days_inclusive():
    assert destination_days({"startDate":"2026-08-01","endDate":"2026-08-03"})==3
