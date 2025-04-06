import streamlit as st
import pandas as pd
import ast

# ====== 1. Data Setup: Load Movies from CSV ======
@st.cache_data  # Cache the data loading to improve performance
def load_movie_data():
    # Read the CSV file
    df = pd.read_csv('movies_metadata.csv')
    
    # Select relevant columns and clean the data
    df = df[['id', 'title', 'genres', 'overview', 'vote_average', 'vote_count']].copy()

    df = df.dropna()  # Remove rows with missing values
    
    # Convert genres from string to list (it's stored as string representation of list)
    df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)] if isinstance(x, str) else [])
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).astype(int)
    
    # Create movies list in the format our app expects
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
    
    # Create lookup dict by title for quick access (case-insensitive keys for matching)
    movie_lookup = {m["title"].lower(): m for m in movies}
    
    return movies, movie_lookup

# Load the data
try:
    movies, movie_lookup = load_movie_data()
except Exception as e:
    st.error(f"Error loading movie data: {str(e)}")
    movies = []
    movie_lookup = {}

# Define synthetic user profiles (each user has a list of liked movie IDs)
user_profiles = {
    "User1": ["603", "27205", "157336"],       # Sci-Fi fan 
    "User2": ["256835", "12"],               # Animation fan
    "User3": ["238", "278", "680"],          # Crime/Drama fan
    "User4": ["603", "680", "155"],          # Mixed tastes
    "User5": ["597", "11036"],               # Romance/Drama fan
    "User6": ["597", "155", "99861"],       # Mixed mainstream
    "User7": ["603", "99861"],            # Sci-Fi & Superhero fan
    "User8": ["155", "99861"],            # Superhero fan
    "User9": ["278"]                     # Drama fan
}
# Note: The user profile IDs have been updated to match actual movie IDs from the dataset

# ====== 2. LLM Configuration (Gemini API) ======
# We use Google Gemini via the google.generativeai SDK. Ensure you've installed it (pip install google-generativeai)
# and set up an API key from Google AI Studio.
try:
    import google.generativeai as genai
    have_genai = True
except ImportError:
    have_genai = False

API_KEY = "AIzaSyB8H_nafmFcSG9cObSxLMZlaPGRnIFTxYc"  # <<< REPLACE with your Gemini API key for full LLM functionality.
if have_genai and API_KEY and API_KEY != "YOUR-API-KEY":
    genai.configure(api_key=API_KEY)
    # Choose a Gemini model to use for text generation.
    # e.g., 'gemini-1.5-pro' (latest large model) or 'gemini-1.5-flash' (faster, smaller).
    model_name = "gemini-2.0-flash"  # using the Pro model for better quality (requires appropriate access tier)
    model = genai.GenerativeModel(model_name)
else:
    model = None

def compute_weighted_score(movie, m, C):
    v = movie.get('vote_count', 0)
    R = movie.get('vote_average', 0)
    return (v / (v + m)) * R + (m / (v + m)) * C

def generate_text(prompt, max_tokens=256, temperature=0.7):
    """Helper to generate text from the LLM given a prompt. Returns the generated text or None."""
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens  # Set your desired maximum number of tokens
    )
    if not model:
        print("[LLM] Model not available. Please configure the Gemini API key.")
        return None  # LLM not available or not configured
    try:
        # Use the Gemini model to generate a completion for the prompt
        response = model.generate_content(prompt, generation_config=generation_config)
        result_text = ""
        if hasattr(response, "text"):
            result_text = response.text
        elif isinstance(response, str):
            result_text = response  # (just in case the API returns raw text string)
        print(f"[LLM] Generated: {result_text}")
        # Strip any leading/trailing whitespace/newlines from result
        return result_text.strip() if result_text else ""
    except Exception as e:
        # If any error occurs during API call, print (or log) it and return None
        print(f"[LLM error] {e}")
        return None

# ====== 3. Recommendation Strategy Implementations ======

# -- 3.1 Content-Based Filtering (Semantic Similarity) --
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Prepare a TF-IDF vectorizer on movie descriptions to simulate content-based embeddings
all_descriptions = [m["description"] for m in movies]
tfidf = TfidfVectorizer().fit(all_descriptions)
desc_matrix = tfidf.transform(all_descriptions)  # matrix of shape (len(movies), vocab_size)

def recommend_content_based(query_text, top_n=5, min_votes=1000):
    """
    Content-based recommendation with weighted rating: find movies whose descriptions best match the query_text.
    Filters for popularity (min_votes) and sorts using weighted rating.
    """
    if not query_text:
        return []

    # Get ID to exclude if query matches a title
    exclude_id = None
    if query_text.lower() in movie_lookup:
        exclude_id = movie_lookup[query_text.lower()]["id"]

    # Create TF-IDF vector for query
    query_vec = tfidf.transform([query_text])
    sims = cosine_similarity(query_vec, desc_matrix)[0]

    # Compute C and m for weighted score
    votes = [m['vote_count'] for m in movies if m['vote_count'] >= min_votes]
    if not votes:
        return []
    C = np.mean([m['vote_average'] for m in movies if m['vote_count'] >= min_votes])
    m = min_votes

    # Gather scored movies
    scored_movies = []
    for idx in np.argsort(sims)[::-1]:
        movie = movies[idx]
        if exclude_id and movie["id"] == exclude_id:
            continue
        if movie.get('vote_count', 0) >= m:
            score = compute_weighted_score(movie, m, C)
            movie['score'] = score
            scored_movies.append(movie)
        if len(scored_movies) >= top_n * 3:  # buffer for better results
            break

    # Sort by weighted score
    sorted_movies = sorted(scored_movies, key=lambda m: m['score'], reverse=True)

    return sorted_movies[:top_n]


# Optionally, we can use the LLM to augment the query for better retrieval (as described in the article).
# For example, generate related keywords or reformulations to expand the query's meaning.
def augment_query_with_llm(query_text):
    """
    Use LLM to augment a user query with related phrases or keywords (LLM-augmented retrieval).
    """
    if not query_text or not model:
        return query_text  # no augmentation if LLM not available
    prompt = f"Generate a few alternative search queries or keywords related to: '{query_text}'."
    augmented = generate_text(prompt, max_tokens=50, temperature=0.3)
    if augmented:
        # Simply append the augmented text to the original query for a broader search
        combined_query = query_text + " " + augmented
        return combined_query
    return query_text

# -- 3.2 Collaborative Filtering (User-Based or Item-Based) --
def recommend_collaborative(query_text, top_n=5):
    """
    Collaborative filtering recommendation: Based on user profiles.
    If query_text corresponds to a known user ID, use that user's likes.
    If it contains movie titles, treat those as the user's liked items.
    Returns a list of recommended movies (excluding ones the user already knows).
    """
    if not query_text:
        return []
    liked_ids = set()
    # Check if query_text matches a known user profile (e.g., "User1")
    if query_text in user_profiles:
        liked_ids = set(user_profiles[query_text])
    else:
        # Otherwise, attempt to parse movie titles from the query text
        text_lower = query_text.lower()
        for title, movie in movie_lookup.items():
            if title in text_lower:
                liked_ids.add(movie["id"])
    print(liked_ids)
    if not liked_ids:
        return []  # no identifiable likes found in the query
    # Gather candidates from other users who liked any of these movies
    candidate_score = {}  # movie_id -> aggregated score (count of like occurrences among similar users)
    for user, liked in user_profiles.items():
        # skip the user if it's the same user (when query_text is a user ID)
        if query_text in user_profiles and user == query_text:
            continue
        # if this user has any of the liked_ids, consider their likes
        if liked_ids.intersection(set(liked)):
            for mid in liked:
                if mid in liked_ids:
                    continue  # skip items the target user already liked
                candidate_score[mid] = candidate_score.get(mid, 0) + 1
    if not candidate_score:
        return []  # no recommendations could be derived (maybe user likes are unique)
    # Sort candidate movies by score (descending) and take top_n
    ranked_candidates = sorted(candidate_score.items(), key=lambda x: x[1], reverse=True)
    recommendations = []
    for mid, score in ranked_candidates[:top_n]:
        # find movie dict by id
        movie = next((m for m in movies if m["id"] == mid), None)
        if movie:
            recommendations.append(movie)
    return recommendations

# -- 3.3 LLM-Enhanced Hybrid Approach (RAG and Re-ranking) --
def recommend_hybrid(query_text, top_n=5):
    """
    Hybrid recommendation: combine content-based and collaborative results, then use the LLM to refine the final suggestions.
    This uses a Retrieval-Augmented Generation approach: we retrieve candidate items, then prompt the LLM (Gemini) to pick the best ones.
    """
    # Step 1: Retrieve initial candidates using content-based and/or collaborative filtering
    candidates = {}
    # Get top content-based matches (using augmented query to improve recall, if LLM available)
    aug_query = augment_query_with_llm(query_text) if query_text else query_text
    for movie in recommend_content_based(aug_query, top_n=top_n*2):  # get more than needed, to have enough candidates
        candidates[movie["id"]] = movie
    # Get top collaborative matches (if query mentions a user or known liked items)
    for movie in recommend_collaborative(query_text, top_n=top_n*2):
        candidates[movie["id"]] = movie
    # If no candidates found via either method, just return empty
    if not candidates:
        return "*(No candidates found for the given query.)*"
    # Limit candidate list size for the LLM prompt (to avoid too long prompt)
    candidate_list = list(candidates.values())[:max(10, top_n*2)]  # at most 10 candidates in prompt
    # Step 2: Prepare a prompt for the LLM to rank/select from these candidates
    prompt_lines = []
    prompt_lines.append(f'User request: "{query_text}"')
    prompt_lines.append("Candidate movies:")
    for i, movie in enumerate(candidate_list, start=1):
        title = movie["title"]
        desc = movie["description"]
        prompt_lines.append(f"{i}. {title}: {desc}")
    prompt_lines.append(f"\nConsidering the user's request and the above candidates, ")
    prompt_lines.append(f"select the {min(top_n, len(candidate_list))} best recommendations for the user.")
    prompt_lines.append("Provide a brief reason for each choice. Format the answer as a numbered list of recommendations.")
    llm_prompt = "\n".join(prompt_lines)
    # Step 3: Call the LLM to get the final ranked recommendations with explanations
    result = generate_text(llm_prompt, max_tokens=300, temperature=0.5)
    if result:
        return result  # a string containing the LLM's recommended list (with reasons)
    else:
        # Fallback: if LLM fails or not available, return a simple list of top candidate titles as a message
        simple_list = [c["title"] for c in candidate_list[:top_n]]
        return " / ".join(simple_list)  # e.g., "Movie1 / Movie2 / Movie3"

# -- 3.4 Few-Shot Prompting (LLM-Only Generative Recommendations) --
def recommend_few_shot(query_text):
    """
    Few-shot prompted recommendation: directly ask the LLM to generate recommendations from scratch.
    We include a few examples in the prompt to demonstrate the desired format and style.
    """
    if not query_text:
        return ""
    # Construct a few-shot prompt with examples
    example_prompt = """You are a movie recommendation assistant. You provide recommendations in a numbered list with a short reason for each.
Example 1:
User: "I love space adventure and science fiction movies."
Assistant:
1. Interstellar - A visually stunning space epic that explores deep emotional themes and scientific concepts.
2. The Martian - A thrilling yet humorous survival story on Mars with a clever protagonist.
3. Guardians of the Galaxy - A fun, action-packed space adventure with a lovable team of misfit heroes.

Example 2:
User: "I'm looking for a classic crime drama film."
Assistant:
1. The Godfather - A legendary mafia drama with powerful performances and storytelling.
2. Pulp Fiction - An iconic crime film with intersecting stories and dark humor.
3. Goodfellas - A gritty, true-to-life gangster saga directed by Martin Scorsese.

Now it's your turn.
User: "{user_query}"
Assistant:\n""".replace("{user_query}", query_text)
    # The prompt includes two examples and then the user's query.
    result = generate_text(example_prompt, max_tokens=200, temperature=0.7)
    if result:
        return result  # return the LLM-generated recommendation list (as a string)
    else:
        return "*No response (LLM not available).*"
    
def recommend_genre_based(query_text, top_n=5, min_votes=1000):
    genres_of_interest = set()

    # Get genres from user profile or query
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

    # Compute C (mean vote) and m (min votes threshold)
    votes = [m['vote_count'] for m in movies if m['vote_count'] >= min_votes]
    if not votes:
        return []
    C = np.mean([m['vote_average'] for m in movies if m['vote_count'] >= min_votes])
    m = min_votes

    # Filter matching movies with enough votes
    matching_movies = [movie for movie in movies if
                       genres_of_interest.intersection(set(movie["genres"])) and
                       movie.get('vote_count', 0) >= m]

    # Compute weighted score
    for movie in matching_movies:
        movie['score'] = compute_weighted_score(movie, m, C)

    # Sort by weighted score
    sorted_movies = sorted(matching_movies, key=lambda m: m['score'], reverse=True)

    return sorted_movies[:top_n]



# ====== 4. Streamlit App UI ======
st.title("🎬 LLM-Powered Recommendation System")
st.write("Enter a movie-related query and select a recommendation strategy to see suggestions.")

# Sidebar controls for strategy selection
strategy = st.sidebar.selectbox(
    "Recommendation Strategy:",
    ["Content-Based", "Collaborative Filtering", "LLM Hybrid (RAG)", "LLM Few-Shot Prompting", "Genre-Based"]
)
st.sidebar.write("## Instructions:")
st.sidebar.write("""
- **Content-Based**: Finds movies similar to your query (by description).
- **Collaborative Filtering**: Finds movies liked by users with similar taste (based on provided movie or user profile).
- **LLM Hybrid (RAG)**: Uses an LLM to refine recommendations from content/collaborative candidates.
- **LLM Few-Shot Prompting**: Directly asks the LLM for recommendations (with minimal guidance).
- **Genre-Based**: Recommends movies based on genres either from user profile or directly mentioned genres.
""")

# Input for query
user_input = st.text_input(
    "Your query (e.g. favorite movies, a description, or a user ID):",
    placeholder="e.g. 'I enjoyed The Matrix and Inception' or 'Family-friendly animated film' or 'User3'"
)
# Button to trigger recommendation
if st.button("Get Recommendations"):
    if not user_input:
        st.warning("Please enter a query to get recommendations.")
    else:
        # Determine which strategy function to call
        if strategy == "Content-Based":
            recs = recommend_content_based(user_input, top_n=5)
            if recs:
                st.subheader("Recommended Movies (Content-Based)")
                for idx, movie in enumerate(recs, start=1):
                    expander = st.expander(f"{idx}. {movie['title']}")
                    expander.write(movie["description"])
            else:
                st.write("No recommendations could be found for your query.")
        elif strategy == "Collaborative Filtering":
            recs = recommend_collaborative(user_input, top_n=5)
            if recs:
                st.subheader("Recommended Movies (Collaborative Filtering)")
                # We can also show why these were recommended (e.g., which liked movies led to them)
                # For simplicity, just list them with descriptions.
                for idx, movie in enumerate(recs, start=1):
                    expander = st.expander(f"{idx}. {movie['title']}")
                    expander.write(movie["description"])
            else:
                st.write("No collaborative recommendations found. Try referencing a known movie title or user ID.")
        elif strategy == "LLM Hybrid (RAG)":
            result_text = recommend_hybrid(user_input, top_n=5)
            st.subheader("LLM-Refined Recommendations")
            if isinstance(result_text, str):
                # If the result is a string (LLM output or fallback string), display it.
                st.markdown(result_text)
            else:
                # In case something else is returned (shouldn't happen here), just show it.
                st.write(result_text)
        elif strategy == "LLM Few-Shot Prompting":
            result_text = recommend_few_shot(user_input)
            st.subheader("LLM-Generated Recommendations")
            if result_text:
                st.markdown(result_text)
            else:
                st.write("No response from LLM. Make sure the LLM API is configured properly.")
        elif strategy == "Genre-Based":
            recs = recommend_genre_based(user_input, top_n=5)
            if recs:
                st.subheader("Recommended Movies (Genre-Based)")
                for idx, movie in enumerate(recs, start=1):
                    expander = st.expander(f"{idx}. {movie['title']} (⭐ {movie['vote_average']})")
                    expander.write(movie["description"])
                    expander.write(f"Genres: {', '.join(movie['genres'])}")
            else:
                st.write("No genre-based recommendations found. Try entering a user ID or a genre name like 'Action'.")

