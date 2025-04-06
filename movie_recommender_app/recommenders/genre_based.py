# recommenders/genre_based.py
import numpy as np
from utils.data_loader import load_movie_data
from recommenders.content_based import compute_weighted_score

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

def recommend_genre_based(query_text, top_n=5, min_votes=1000):
    genres_of_interest = set()

    if query_text in user_profiles:
        liked_ids = set(user_profiles[query_text])
        for movie in movies:
            if movie["id"] in liked_ids:
                genres_of_interest.update(movie["genres"])
    else:
        all_genres = {genre for movie in movies for genre in movie["genres"]}
        for genre in all_genres:
            if genre.lower() in query_text.lower():
                genres_of_interest.add(genre)

    if not genres_of_interest:
        return []

    votes = [m['vote_count'] for m in movies if m['vote_count'] >= min_votes]
    if not votes:
        return []
    C = np.mean([m['vote_average'] for m in movies if m['vote_count'] >= min_votes])
    m = min_votes

    matching = [movie for movie in movies if genres_of_interest.intersection(set(movie["genres"])) and movie['vote_count'] >= m]

    for movie in matching:
        movie['score'] = compute_weighted_score(movie, m, C)

    return sorted(matching, key=lambda m: m['score'], reverse=True)[:top_n]
