# utils/data_loader.py
import pandas as pd
import ast
import streamlit as st

@st.cache_data
def load_movie_data():
    df = pd.read_csv('movies_metadata.csv')
    df = df[['id', 'title', 'genres', 'overview', 'vote_average', 'vote_count']].copy()
    df = df.dropna()

    df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)] if isinstance(x, str) else [])
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).astype(int)

    movies = []
    for _, row in df.iterrows():
        movies.append({
            "id": str(row['id']),
            "title": row['title'],
            "genres": row['genres'],
            "description": row['overview'],
            "vote_average": row['vote_average'],
            "vote_count": row['vote_count']
        })

    movie_lookup = {m["title"].lower(): m for m in movies}

    return movies, movie_lookup
