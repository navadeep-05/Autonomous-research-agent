import re
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

# Maximum evidence items to send for gap analysis
MAX_EVIDENCE_FOR_GAPS = 12


def parse_gap_queries(response_text):
    """Extract at most one actionable search query, ignoring response labels."""
    text = str(response_text or "").strip()
    normalized = re.sub(r"[\s_]+", "", text).upper()
    if not text or normalized in {"NOGAPS", "NO_GAPS"}:
        return []

    queries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        line = re.sub(
            r"^(?:search\s+query\s+to\s+fill\s+(?:the\s+)?gap|gap\s+query)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if not line or line.casefold() in {"no gaps", "no_gaps"}:
            continue
        line = re.sub(r"^(?:query|search query)\s*:\s*", "", line, flags=re.I).strip()
        line = line.strip(" \t\"'“”‘’`")
        if line and line.casefold() not in {"no gaps", "no_gaps"}:
            queries.append(line)
    return list(dict.fromkeys(queries))[:1]


def detect_gaps(question, evidence):
    """
    Identify important information that is still missing.
    Returns a list of gap queries, or empty list if no gaps found.
    Never raises.
    """

    # Cap evidence to keep prompt within token limits
    capped = evidence[:MAX_EVIDENCE_FOR_GAPS]

    # Convert evidence list into readable text
    evidence_text = "\n".join(
        f"- {str(item)[:200]}" for item in capped
    )

    try:
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
- ONLY if a major, critical aspect is completely unaddressed and essential to answering the question, return a maximum of 1 specific search query.
- Return either NO_GAPS or the query text only. Do not add labels, quotes, or explanations.
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
            max_completion_tokens=300
        )

        return parse_gap_queries(response.choices[0].message.content)

    except Exception as e:
        print(f"[gap_detector] Error: {e}")
        return []