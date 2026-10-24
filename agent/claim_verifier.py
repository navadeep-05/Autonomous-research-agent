import json
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

# Keep the verifier prompt bounded while including original source passages.
MAX_EVIDENCE_FOR_VERIFY = 10
MAX_SOURCES_FOR_VERIFY = 24
MAX_SOURCE_CHARS = 3000


def parse_verification_results(
    response_text,
    evidence,
    sources,
    *,
    allowed_source_ids=None,
):
    """Validate verifier labels and bind supported results to exact source quotes."""
    text = str(response_text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Verifier response must be a JSON list.")

    sources_by_id = {
        source.get("source_id"): source
        for source in sources
        if source.get("source_id")
    }
    allowed_source_ids = allowed_source_ids or [[] for _ in evidence]
    rows_by_index = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        index = row.get("evidence_index")
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(evidence):
            rows_by_index[index] = row

    results = []
    missing = False
    for index, item in enumerate(evidence):
        row = rows_by_index.get(index)
        if row is None:
            missing = True
            results.append({
                "evidence_index": index,
                "claim": str(item),
                "status": "UNVERIFIED",
                "allowed_source_ids": (
                    allowed_source_ids[index]
                    if index < len(allowed_source_ids)
                    else []
                ),
                "source_id": None,
                "quote": "",
            })
            continue

        status = str(row.get("status", "")).upper()
        if status not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNKNOWN"}:
            status = "UNVERIFIED"
        source_id = row.get("source_id")
        if not isinstance(source_id, str):
            source_id = None
        allowed_ids = (
            allowed_source_ids[index]
            if index < len(allowed_source_ids)
            else []
        )
        if source_id not in allowed_ids:
            source_id = None
        source = sources_by_id.get(source_id)
        quote = str(row.get("quote") or "").strip()
        quote_is_verbatim = bool(
            source is not None
            and quote
            and quote in str(source.get("content", ""))
        )

        if status in {"SUPPORTED", "PARTIAL"} and not quote_is_verbatim:
            status = "UNVERIFIED"
            source_id = None
            quote = ""
        elif source is None or (quote and not quote_is_verbatim):
            source_id = None
            quote = ""

        results.append({
            "evidence_index": index,
            "claim": str(item),
            "status": status,
            "allowed_source_ids": allowed_ids,
            "source_id": source_id,
            "quote": quote,
        })

    if missing:
        results[0]["verification_warning"] = "Verifier omitted one or more evidence items."
    return results


def verify_claims(
    evidence,
    sources=None,
    *,
    allowed_source_ids=None,
    raise_on_error=False,
):
    """
    Assess each extracted evidence item against the original retrieved sources.
    """
    evidence = list(evidence or [])[:MAX_EVIDENCE_FOR_VERIFY]
    sources = list(sources or [])[:MAX_SOURCES_FOR_VERIFY]
    request_payload = {
        "evidence": [
            {
                "evidence_index": index,
                "statement": str(item)[:300],
                "allowed_source_ids": (
                    allowed_source_ids[index]
                    if allowed_source_ids and index < len(allowed_source_ids)
                    else []
                ),
            }
            for index, item in enumerate(evidence)
        ],
        "sources": [
            {
                "source_id": source.get("source_id", ""),
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": str(source.get("content", ""))[:MAX_SOURCE_CHARS],
            }
            for source in sources
        ],
    }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a source-grounded claim verifier. For every supplied evidence item, assess
whether the statement is supported by the ORIGINAL SOURCE CONTENT. The evidence
statements are candidates, not proof.

Return ONLY a JSON array, with one object per evidence item:
{"evidence_index": 0, "status": "SUPPORTED|PARTIAL|UNSUPPORTED|UNKNOWN",
 "source_id": "one supplied source ID or null", "quote": "exact verbatim source quote or empty string"}

Use SUPPORTED only when a supplied passage directly supports the whole statement.
Use PARTIAL when a passage supports only part. Use UNSUPPORTED only after checking
the supplied passages and finding the statement contradicted or unsupported; use
UNKNOWN if the passages are insufficient. Never invent a statement, source ID, or
quote. For SUPPORTED and PARTIAL, provide the exact quote and its source ID.
Each evidence item's source_id MUST come from that item's allowed_source_ids;
never use a source supplied for another evidence item.
"""
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False)
                }
            ],
            max_completion_tokens=1800
        )

        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[claim_verifier] Error: {exc}")
        if raise_on_error:
            raise
        return ""