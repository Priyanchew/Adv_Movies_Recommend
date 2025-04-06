# recommenders/content_based.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm.llm_helper import generate_text
from utils.data_loader import load_movie_data

movies, movie_lookup = load_movie_data()
all_descriptions = [m["description"] for m in movies]
tfidf = TfidfVectorizer().fit(all_descriptions)
desc_matrix = tfidf.transform(all_descriptions)

def compute_weighted_score(movie, m, C):
    v = movie.get('vote_count', 0)
    R = movie.get('vote_average', 0)
    return (v / (v + m)) * R + (m / (v + m)) * C

def recommend_content_based(query_text, top_n=5, min_votes=1000):
    if not query_text:
        return []

    exclude_id = movie_lookup.get(query_text.lower(), {}).get("id")
    query_vec = tfidf.transform([query_text])
    sims = cosine_similarity(query_vec, desc_matrix)[0]

    votes = [m['vote_count'] for m in movies if m['vote_count'] >= min_votes]
    if not votes:
        return []
    C = np.mean([m['vote_average'] for m in movies if m['vote_count'] >= min_votes])
    m = min_votes

    scored_movies = []
    for idx in np.argsort(sims)[::-1]:
        movie = movies[idx]
        if exclude_id and movie["id"] == exclude_id:
            continue
        if movie['vote_count'] >= m:
            score = compute_weighted_score(movie, m, C)
            movie['score'] = score
            scored_movies.append(movie)
        if len(scored_movies) >= top_n * 3:
            break

    return sorted(scored_movies, key=lambda m: m['score'], reverse=True)[:top_n]

def augment_query_with_llm(query_text):
    if not query_text:
        return query_text
    prompt = f"Generate alternative search queries or keywords for: '{query_text}'"
    augmented = generate_text(prompt, max_tokens=50, temperature=0.3)
    return f"{query_text} {augmented}" if augmented else query_text
