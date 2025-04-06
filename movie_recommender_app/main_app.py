
# main_app.py
import streamlit as st
from utils.data_loader import load_movie_data
from recommenders.content_based import recommend_content_based, augment_query_with_llm
from recommenders.collaborative import recommend_collaborative
from recommenders.hybrid import recommend_hybrid
from recommenders.few_shot import recommend_few_shot
from recommenders.genre_based import recommend_genre_based

st.title("LLM-Powered Recommendation System")
st.write("Enter a movie-related query and select a recommendation strategy to see suggestions.")

# Load data
movies, movie_lookup = load_movie_data()

# Sidebar controls
strategy = st.sidebar.selectbox("Recommendation Strategy:", [
    "Content-Based", "Collaborative Filtering", "LLM Hybrid (RAG)", "LLM Few-Shot Prompting", "Genre-Based"
])

st.sidebar.write("## Instructions:")
st.sidebar.write("""
- **Content-Based**: Finds movies similar to your query (by description).
- **Collaborative Filtering**: Finds movies liked by users with similar taste.
- **LLM Hybrid (RAG)**: Uses an LLM to refine recommendations from content/collaborative candidates.
- **LLM Few-Shot Prompting**: Directly asks the LLM for recommendations.
- **Genre-Based**: Recommends movies based on genres.
""")

user_input = st.text_input("Your query:", placeholder="e.g. 'I enjoyed The Matrix and Inception' or 'User3'")

if st.button("Get Recommendations"):
    if not user_input:
        st.warning("Please enter a query to get recommendations.")
    else:
        if strategy == "Content-Based":
            recs = recommend_content_based(user_input, top_n=5)
            st.subheader("Recommended Movies (Content-Based)")
            for idx, movie in enumerate(recs, start=1):
                expander = st.expander(f"{idx}. {movie['title']}")
                expander.write(movie["description"])

        elif strategy == "Collaborative Filtering":
            recs = recommend_collaborative(user_input, top_n=5)
            st.subheader("Recommended Movies (Collaborative Filtering)")
            for idx, movie in enumerate(recs, start=1):
                expander = st.expander(f"{idx}. {movie['title']}")
                expander.write(movie["description"])

        elif strategy == "LLM Hybrid (RAG)":
            result = recommend_hybrid(user_input, top_n=5)
            st.subheader("LLM-Refined Recommendations")
            st.markdown(result)

        elif strategy == "LLM Few-Shot Prompting":
            result = recommend_few_shot(user_input)
            st.subheader("LLM-Generated Recommendations")
            st.markdown(result)

        elif strategy == "Genre-Based":
            recs = recommend_genre_based(user_input, top_n=5)
            st.subheader("Recommended Movies (Genre-Based)")
            for idx, movie in enumerate(recs, start=1):
                expander = st.expander(f"{idx}. {movie['title']} (\u2b50 {movie['vote_average']})")
                expander.write(movie["description"])
                expander.write(f"Genres: {', '.join(movie['genres'])}")