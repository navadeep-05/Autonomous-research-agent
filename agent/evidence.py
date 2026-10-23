import time
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

# Maximum source snippets and chars per snippet to stay within token limits
MAX_SOURCES = 4
MAX_CHARS_PER_SOURCE = 800


def extract_evidence(task, search_result):
    """
    Extract important factual evidence from a search result.
    Returns a list of strings. Never raises.
    """

    entries = (search_result if isinstance(search_result, list)
               else [{"url": "", "content": str(search_result)}])
    # Cap to avoid token overflow
    entries = entries[:MAX_SOURCES]

    user_content = (
        f"Research task:\n{task}\n\nResearch result:\n"
        + "\n\n".join(
            f"[{r.get('url', '')}]\n{str(r.get('content', ''))[:MAX_CHARS_PER_SOURCE]}"
            for r in entries
        )
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """
You are an evidence extraction agent.

Extract the most important factual claims from
the provided research result.

Return a numbered list of up to 8 items.
Do not add information that is not present
in the research result.
"""
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                max_completion_tokens=600
            )

            # Get the extracted evidence
            evidence_text = response.choices[0].message.content
            if not evidence_text:
                return []

            # Convert text into a list
            evidence = []

            for line in evidence_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Remove numbering
                claim = line.lstrip("0123456789.-) ").strip()
                if claim:
                    evidence.append(claim)

            return evidence

        except Exception as e:
            print(f"[evidence] Attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)  # Brief pause before retry

    return []