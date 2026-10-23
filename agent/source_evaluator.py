from groq import Groq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-20b"


def evaluate_source(source_text):
    """
    Evaluate the reliability characteristics of a source.
    Returns evaluation string. Never raises.
    """
    # Cap input to prevent token overflow
    source_text = str(source_text)[:1500]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a research source evaluation agent.

Evaluate the provided source using these factors:

1. Authority
2. Relevance
3. Recency
4. Evidence quality

Return a short structured evaluation (4 lines max).

Do not invent information that is not present
in the provided source.
"""
                },
                {
                    "role": "user",
                    "content": source_text
                }
            ],
            max_completion_tokens=300
        )

        return response.choices[0].message.content or "Evaluation unavailable."
    except Exception as e:
        print(f"[source_evaluator] Error: {e}")
        return "Evaluation unavailable due to processing error."