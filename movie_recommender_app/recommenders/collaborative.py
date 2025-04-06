# recommenders/collaborative.py
from utils.data_loader import load_movie_data

movies, movie_lookup = load_movie_data()

user_profiles = {
    "User1": ["603", "27205", "157336"],
    "User2": ["256835", "12"],
    "User3": ["238", "278", "680"],
    "User4": ["603", "680", "155"],
    "User5": ["597", "11036"],
    "User6": ["597", "155", "99861"],
    "User7": ["603", "99861"],
    "User8": ["155", "99861"],
    "User9": ["278"]
}

def recommend_collaborative(query_text, top_n=5):
    if not query_text:
        return []

    liked_ids = set()
    if query_text in user_profiles:
        liked_ids = set(user_profiles[query_text])
    else:
        for title, movie in movie_lookup.items():
            if title in query_text.lower():
                liked_ids.add(movie["id"])

    if not liked_ids:
        return []

    candidate_score = {}
    for user, liked in user_profiles.items():
        if query_text in user_profiles and user == query_text:
            continue
        if liked_ids.intersection(set(liked)):
            for mid in liked:
                if mid not in liked_ids:
                    candidate_score[mid] = candidate_score.get(mid, 0) + 1

    ranked = sorted(candidate_score.items(), key=lambda x: x[1], reverse=True)
    recommendations = [next((m for m in movies if m["id"] == mid), None) for mid, _ in ranked[:top_n]]

    return [m for m in recommendations if m]
