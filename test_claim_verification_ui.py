from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_empty_citations_render_unverified_evidence_with_source_context():
    evidence = [f"Extracted research fact {index}." for index in range(1, 9)]
    at = AppTest.from_file("app.py")
    at.session_state["result"] = {
        "report": "Research report",
        "plan": ["Research the subject"],
        "evidence": evidence,
        "verified_claims": [],
        "verification_error": "The verifier returned no claims.",
        "citations": [],
        "sources": [
            {
                "source_id": "SRC-001",
                "url": "https://example.com/research",
                "title": "Research source",
                "content": "Source passage that may help assess the extracted facts.",
            }
        ],
        "evidence_matches": [
            {"evidence": fact, "source_id": "SRC-001", "score": 0.42}
            for fact in evidence
        ],
        "gaps": [],
        "iteration": 0,
    }

    at.run(timeout=60)

    assert not at.exception, at.exception
    labels = [tab.label for tab in at.tabs]
    assert len(labels) == 5
    assert not any("Verified Sources" in label for label in labels)

    rendered = "\n".join(item.value for item in at.markdown)
    alerts = "\n".join(
        item.value for item in (*at.warning, *at.error)
    )
    expanders = "\n".join(item.label for item in at.expander)
    assert "Claim Verification & Semantic Source Alignment" in rendered
    assert "The verifier returned no claims." in alerts
    assert "No claim citations were produced" in alerts
    assert "Extracted evidence — UNVERIFIED" in rendered
    for fact in evidence:
        assert fact in rendered
    assert "UNVERIFIED" in rendered
    assert "SRC-001" in rendered
    assert "0.42" in rendered
    assert "View closest source passage" in expanders


def test_cited_claim_renders_status_match_score_and_excerpt():
    at = AppTest.from_file("app.py")
    at.session_state["result"] = {
        "report": "Research report",
        "plan": [],
        "evidence": ["A claim supported by the research."],
        "verified_claims": ["1. A claim supported by the research. [SUPPORTED]"],
        "verification_error": "",
        "citations": [
            {
                "claim": "1. A claim supported by the research. [SUPPORTED]",
                "source_id": "SRC-001",
                "score": 0.81,
                "quote": "This exact sentence appears in the source.",
            }
        ],
        "sources": [
            {
                "source_id": "SRC-001",
                "url": "https://example.com/research",
                "title": "Research source",
                "content": "This exact sentence appears in the source. Relevant source passage.",
            }
        ],
        "gaps": [],
        "iteration": 0,
    }

    at.run(timeout=60)

    assert not at.exception, at.exception
    rendered = "\n".join(item.value for item in at.markdown)
    expanders = "\n".join(item.label for item in at.expander)
    assert "A claim supported by the research." in rendered
    assert "SUPPORTED" in rendered
    assert "Closest source match" in rendered
    assert "0.81" in rendered
    assert "This exact sentence appears in the source." in rendered
    assert "View matched source excerpt" in expanders


def test_verifier_exception_is_recorded_without_upgrading_evidence():
    from agent.graph import verify_node

    with patch("agent.graph.verify_claims", side_effect=RuntimeError("provider unavailable")):
        result = verify_node({"evidence": ["An extracted fact."]})

    assert result["verified_claims"] == []
    assert "verification failed" in result["verification_error"].lower()
