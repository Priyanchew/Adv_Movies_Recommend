# recommenders/hybrid.py
from recommenders.content_based import recommend_content_based, augment_query_with_llm
from recommenders.collaborative import recommend_collaborative
from llm.llm_helper import generate_text


def recommend_hybrid(query_text, top_n=5):
    candidates = {}
    aug_query = augment_query_with_llm(query_text) if query_text else query_text

    for movie in recommend_content_based(aug_query, top_n=top_n * 2):
        candidates[movie["id"]] = movie

    for movie in recommend_collaborative(query_text, top_n=top_n * 2):
        candidates[movie["id"]] = movie

    if not candidates:
        return "*(No candidates found for the given query.)*"

    candidate_list = list(candidates.values())[:max(10, top_n * 2)]
    prompt_lines = [
        f'User request: "{query_text}"',
        "Candidate movies:",
    ]
    for i, movie in enumerate(candidate_list, start=1):
        prompt_lines.append(f"{i}. {movie['title']}: {movie['description']}")

    prompt_lines.append(f"\nSelect the {min(top_n, len(candidate_list))} best recommendations for the user.")
    prompt_lines.append("Provide a brief reason for each. Format as a numbered list.")

    llm_prompt = "\n".join(prompt_lines)
    result = generate_text(llm_prompt, max_tokens=300, temperature=0.5)

    return result if result else " / ".join([m["title"] for m in candidate_list[:top_n]])
