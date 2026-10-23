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

# Maximum evidence items to send (keeps prompt within token budget)
MAX_EVIDENCE_FOR_VERIFY = 10


def verify_claims(evidence):
    """
    Check whether important claims are supported by evidence.
    """

    # Cap evidence to avoid exceeding token limits
    capped = evidence[:MAX_EVIDENCE_FOR_VERIFY]

    # Convert evidence into readable text
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
You are a precise, objective claim verification auditor.

Review the provided research evidence and extract the key factual claims made.

For each claim:
1. State the exact claim clearly and concisely (one sentence).
2. Label its status at the end of the line: [SUPPORTED], [PARTIALLY SUPPORTED], or [UNSUPPORTED].

Requirements:
- Return ONLY a numbered list (e.g. "1. LangGraph uses cyclic graphs. [SUPPORTED]").
- Maximum 8 claims.
- Only classify as [SUPPORTED] if explicitly substantiated by the provided evidence.
- Do NOT generate opinions or extrapolate beyond the evidence.
"""
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{evidence_text}"
                }
            ],
            max_completion_tokens=600
        )

        # Get verification results
        result = response.choices[0].message.content
        return result or ""
    except Exception as e:
        print(f"[claim_verifier] Error: {e}")
        return ""