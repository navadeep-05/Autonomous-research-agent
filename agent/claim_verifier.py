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

def verify_claims(evidence):
    """
    Check whether important claims are supported by evidence.
    """

    # Convert evidence into readable text
    evidence_text = "\n".join(
        f"- {item}" for item in evidence
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a precise, objective claim verification auditor.

Review the provided research evidence and extract the key factual claims made.

For each claim:
1. State the exact claim clearly and concisely.
2. Verify against the evidence and label its status at the end of the line: [SUPPORTED], [PARTIALLY SUPPORTED], or [UNSUPPORTED].

Requirements:
- Return ONLY a numbered list of claims (e.g. "1. LangGraph represents workflows as cyclic graphs. [SUPPORTED]").
- Only classify as [SUPPORTED] if explicitly substantiated by the provided evidence text.
- Do NOT generate ungrounded opinions or extrapolate beyond the evidence.
"""
            },
            {
                "role": "user",
                "content": f"""
Evidence:

{evidence_text}
"""
            }
        ],
        max_completion_tokens=700
    )

    # Get verification results
    result = response.choices[0].message.content

    return result