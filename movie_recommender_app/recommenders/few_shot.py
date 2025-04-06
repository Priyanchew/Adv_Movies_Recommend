# recommenders/few_shot.py
from llm.llm_helper import generate_text

def recommend_few_shot(query_text):
    if not query_text:
        return ""

    example_prompt = f"""You are a movie recommendation assistant. Provide recommendations in a numbered list with reasons.

Example 1:
User: "I love space adventure and science fiction movies."
Assistant:
1. Interstellar - A visually stunning space epic.
2. The Martian - A thrilling survival story on Mars.
3. Guardians of the Galaxy - A fun, action-packed space adventure.

Example 2:
User: "I'm looking for a classic crime drama film."
Assistant:
1. The Godfather - A legendary mafia drama.
2. Pulp Fiction - An iconic crime film with dark humor.
3. Goodfellas - A gritty gangster saga.

Now it's your turn.
User: "{query_text}"
Assistant:
"""
    result = generate_text(example_prompt, max_tokens=200, temperature=0.7)
    return result if result else "*No response (LLM not available).*"
