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

# Caps to keep prompt within token budget
MAX_EVIDENCE = 12
MAX_SOURCES = 10
MAX_CHARS_PER_SOURCE = 250


def generate_report(question, evidence, verified_claims, sources):
    """
    Generate the final research report.
    Returns report string. Never raises.
    """

    # Convert evidence into concise readable text (capped)
    evidence_text = "\n".join(
        f"- {str(item)[:200]}" for item in evidence[:MAX_EVIDENCE]
    )

    # Convert source information into concise snippets
    source_text = "\n".join(
        f"{item['source_id']}: {str(item.get('content', ''))[:MAX_CHARS_PER_SOURCE]}"
        for item in sources[:MAX_SOURCES]
    )

    # Cap verified claims text
    claims_text = str(verified_claims)[:2000] if verified_claims else "No verified claims available."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a rigorous, objective academic research report generation agent.

Synthesize a comprehensive, trustworthy research report based strictly and solely on the provided evidence, verified claims, and sources.

Core Requirements:
1. Anti-Hallucination & Grounding: Every substantive fact MUST be directly derived from the provided evidence. Do NOT invent facts, statistics, or sources.
2. Fairness & Neutrality: Present a balanced, unbiased view.
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
{claims_text}

Sources:
{source_text}
"""
                }
            ],
            max_completion_tokens=1500
        )

        # Return generated report
        result = response.choices[0].message.content
        return result or "Report generation returned empty response."

    except Exception as e:
        print(f"[report_generator] Error: {e}")
        # Return a minimal report with the available evidence
        fallback = f"# Research Report: {question}\n\n"
        fallback += "## Evidence Summary\n\n"
        for item in evidence[:MAX_EVIDENCE]:
            fallback += f"- {str(item)[:200]}\n"
        fallback += "\n\n*Note: Full report generation failed. Above is the raw evidence collected.*"
        return fallback