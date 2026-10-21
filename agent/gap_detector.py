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

def detect_gaps(question, evidence):
    """
    Identify important information that is still missing.
    """

    # Convert evidence list into readable text
    evidence_text = "\n".join(
        f"- {item}" for item in evidence
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an efficient, objective research gap detection auditor.

Analyze the user's research question against the collected evidence.

Decision Criteria:
- If the collected evidence already provides a reasonable, factual, and informative answer to the core question, you MUST return:
NO_GAPS
- ONLY if a major, critical aspect is completely unaddressed and essential to answering the question, return a maximum of 1 or 2 specific search queries.
- Do NOT invent pedantic, trivial, or minor academic gaps. Value speed and conciseness.
"""
            },
            {
                "role": "user",
                "content": f"""
Research question:
{question}

Collected evidence:
{evidence_text}
"""
            }
        ],
        max_completion_tokens=500
    )

    result = response.choices[0].message.content.strip()

    # Stop if the research appears complete
    if "NOGAPS" in result.upper().replace(" ", "").replace("_", ""):
        return []

    gaps = []

    # Convert numbered output into a Python list
    for line in result.splitlines():

        line = line.strip()

        if not line:
            continue

        gap = line.lstrip("0123456789.-) ").strip()

        if gap:
            gaps.append(gap)

    return gaps