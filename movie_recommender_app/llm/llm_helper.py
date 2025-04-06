from dotenv import load_dotenv
import os

try:
    import google.generativeai as genai
    have_genai = True
except ImportError:
    have_genai = False

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if have_genai and API_KEY and API_KEY != "YOUR-API-KEY":
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

def generate_text(prompt, max_tokens=256, temperature=0.7):
    if not model:
        print("[LLM] Model not available.")
        return None
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, str):
            return response.strip()
        return ""
    except Exception as e:
        print(f"[LLM error] {e}")
        return None