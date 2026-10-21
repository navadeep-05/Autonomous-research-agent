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

def generate_report(question, evidence, verified_claims, sources):
    """
    Generate the final research report.
    """

    # Convert evidence into concise readable text (cap top 15 facts to avoid token bloat)
    evidence_text = "\n".join(
        f"- {item}" for item in evidence[:15]
    )

    # Convert source information into concise snippets
    source_text = "\n".join(
        f"{item['source_id']}: {str(item.get('content', ''))[:300]}"
        for item in sources[:12]
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a rigorous, objective academic research report generation agent.

Synthesize a comprehensive, trustworthy research report based strictly and solely on the provided evidence, verified claims, and sources.

Core Requirements:
1. Anti-Hallucination & Grounding: Every substantive fact, assertion, or number MUST be directly derived from the provided evidence. Do NOT invent facts, statistics, or sources.
2. Fairness & Neutrality: Present a balanced, unbiased view covering multiple perspectives, nuances, trade-offs, and limitations.
3. Strict Inline Citations: Cite every factual point using exact bracketed source identifiers like [SRC-001].
4. Report Structure:
   # <Title>
   ## 1. Executive Summary
   ## 2. Key Findings & Supported Evidence
   ## 3. In-Depth Analysis & Discussion
   ## 4. Nuances, Limitations & Remaining Gaps
   ## 5. Conclusion
   ## 6. References
"""
            },
            {
                "role": "user",
                "content": f"""
Research Question:
{question}

Evidence:
{evidence_text}

Verified Claims:
{verified_claims}

Sources:
{source_text}
"""
            }
        ],
        max_completion_tokens=1500
    )

    # Return generated report
    return response.choices[0].message.content