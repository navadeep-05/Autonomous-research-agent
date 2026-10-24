import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from agent.claim_verifier import parse_verification_results, verify_claims
from agent.gap_detector import detect_gaps, parse_gap_queries
from agent.graph import (
    MAX_GAP_ROUNDS,
    MAX_TASKS,
    citations_node,
    plan_node,
    report_node,
    research_one,
    verify_node,
)
from RAG.retriever import retrieve as kb_retrieve
from RAG.vector_store import VectorStore


def test_verifier_receives_original_source_and_quote_is_validated():
    evidence = ["The device reached 12 watts."]
    sources = [{
        "source_id": "SRC-001",
        "title": "Device report",
        "url": "https://example.com/device",
        "content": "The device reached 12 watts during the test.",
    }]
    response_json = json.dumps([{
        "evidence_index": 0,
        "status": "SUPPORTED",
        "source_id": "SRC-001",
        "quote": "The device reached 12 watts",
    }])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=response_json))]
    )

    with patch("agent.claim_verifier.client.chat.completions.create", return_value=response) as create:
        raw_response = verify_claims(
            evidence,
            sources,
            allowed_source_ids=[["SRC-001"]],
            raise_on_error=True,
        )

    request_content = create.call_args.kwargs["messages"][1]["content"]
    assert '"allowed_source_ids": ["SRC-001"]' in request_content
    assert sources[0]["content"] in request_content
    assert "The device reached 12 watts." in request_content
    results = parse_verification_results(
        raw_response,
        evidence,
        sources,
        allowed_source_ids=[["SRC-001"]],
    )
    assert results[0]["status"] == "SUPPORTED"
    assert results[0]["source_id"] == "SRC-001"
    assert results[0]["quote"] == "The device reached 12 watts"
    assert results[0]["allowed_source_ids"] == ["SRC-001"]

    forged_quote = json.dumps([{
        "evidence_index": 0,
        "status": "SUPPORTED",
        "source_id": "SRC-001",
        "quote": "A quote absent from the source",
    }])
    invalid = parse_verification_results(forged_quote, evidence, sources)
    assert invalid[0]["status"] == "UNVERIFIED"
    assert invalid[0]["source_id"] is None


def test_graph_verification_keeps_source_bound_status_and_citation():
    evidence = ["The device reached 12 watts."]
    source = {
        "source_id": "SRC-001",
        "title": "Device report",
        "url": "https://example.com/device",
        "content": "The device reached 12 watts during the test.",
    }
    origin = source["url"]
    raw = json.dumps([{
        "evidence_index": 0,
        "status": "SUPPORTED",
        "source_id": "SRC-001",
        "quote": "The device reached 12 watts",
    }])
    with patch("agent.graph.verify_claims", return_value=raw) as verify:
        verified = verify_node({
            "evidence": evidence,
            "evidence_origins": [origin],
            "sources": [source],
        })
    verify.assert_called_once_with(
        evidence,
        [source],
        allowed_source_ids=[["SRC-001"]],
        raise_on_error=True,
    )
    assert verified["claim_verifications"][0]["status"] == "SUPPORTED"
    assert verified["verified_claims"] == ["The device reached 12 watts. [SUPPORTED]"]

    state = {
        "verified_claims": verified["verified_claims"],
        "claim_verifications": verified["claim_verifications"],
        "sources": [source],
        "evidence": evidence,
        "evidence_origins": [origin],
    }
    with patch("agent.graph.embed", return_value=np.array([[1.0, 0.0]], dtype="float32")):
        citations = citations_node(state)
    citation = citations["citations"][0]
    assert citation["status"] == "SUPPORTED"
    assert citation["source_id"] == "SRC-001"
    assert citation["quote"] == "The device reached 12 watts"
    assert citation["score"] == 1.0


def test_selected_verifier_source_below_similarity_cutoff_is_not_cited():
    source = {
        "source_id": "SRC-001",
        "title": "Cambridge dictionary",
        "url": "https://dictionary.cambridge.org/",
        "content": "A dictionary lists words and their meanings.",
    }
    state = {
        "claim_verifications": [{
            "evidence_index": 0,
            "claim": "A hotel API supports booking records.",
            "status": "SUPPORTED",
            "source_id": "SRC-001",
            "allowed_source_ids": ["SRC-001"],
            "quote": "A dictionary lists words and their meanings.",
        }],
        "sources": [source],
        "evidence": ["A hotel API supports booking records."],
        "evidence_origins": [source["url"]],
    }
    with patch(
        "agent.graph.embed",
        side_effect=[
            np.array([[1.0, 0.0]], dtype="float32"),
            np.array([[0.05, 0.9987]], dtype="float32"),
            np.array([[1.0, 0.0]], dtype="float32"),
        ],
    ):
        result = citations_node(state)
    citation = result["citations"][0]
    assert citation["source_id"] is None
    assert citation["status"] == "UNVERIFIED"
    assert citation["verifier_status"] == "SUPPORTED"
    assert citation["score"] == 0.05
    assert citation["quote"] == "A dictionary lists words and their meanings."
    assert citation["origin_source_id"] == "SRC-001"


def test_verifier_cannot_attribute_evidence_to_an_unrelated_source():
    sources = [
        {
            "source_id": "SRC-001",
            "url": "https://api.example/hotel",
            "content": "Hotel API records reservations.",
        },
        {
            "source_id": "SRC-002",
            "url": "https://dictionary.example/",
            "content": "A dictionary lists words and their meanings.",
        },
    ]
    evidence = ["A hotel API records reservations."]
    model_response = json.dumps([{
        "evidence_index": 0,
        "status": "SUPPORTED",
        "source_id": "SRC-002",
        "quote": "A dictionary lists words and their meanings.",
    }])
    with patch("agent.graph.verify_claims", return_value=model_response) as verify:
        result = verify_node({
            "evidence": evidence,
            "evidence_origins": ["https://api.example/hotel"],
            "sources": sources,
        })
    verify.assert_called_once_with(
        evidence,
        sources,
        allowed_source_ids=[["SRC-001"]],
        raise_on_error=True,
    )
    assert result["claim_verifications"][0]["status"] == "UNVERIFIED"
    assert result["claim_verifications"][0]["source_id"] is None


def test_research_extraction_keeps_each_claim_bound_to_its_source():
    search_results = [
        {"title": "API docs", "url": "https://api.example/docs", "content": "API facts"},
        {"title": "Dictionary", "url": "https://dictionary.example/", "content": "Word definitions"},
    ]
    with (
        patch("agent.graph.search_web", return_value=search_results),
        patch("agent.graph.kb_retrieve", return_value=[]),
        patch(
            "agent.graph.extract_evidence",
            side_effect=[["API evidence"], ["Dictionary evidence"]],
        ) as extract,
        patch("agent.graph.evaluate_source", return_value="evaluated"),
    ):
        result = research_one("Relevant research query")
    assert extract.call_count == 2
    assert result[1] == ["API evidence", "Dictionary evidence"]
    assert result[2] == [
        "https://api.example/docs",
        "https://dictionary.example/",
    ]


def test_plan_is_capped_at_three_actual_subtasks():
    generated = [
        "Research measured performance",
        "Check independent replication",
        "Review limitations",
        "Ignore this excess task",
    ]
    with patch("agent.graph.create_research_plan", return_value=generated):
        result = plan_node({"question": "Assess the device"})

    assert MAX_TASKS == 3
    assert result["plan"] == generated[:3]


def test_three_named_subtasks_are_rendered_in_the_plan_section():
    tasks = [
        "Measure current device performance",
        "Compare independent evaluations",
        "Identify limitations and uncertainty",
    ]
    at = AppTest.from_file("app.py")
    at.session_state["result"] = {
        "report": "Report",
        "plan": tasks,
        "evidence": [],
        "verified_claims": [],
        "citations": [],
        "sources": [],
        "gaps": [],
        "iteration": 0,
    }
    at.run(timeout=60)

    assert not at.exception, at.exception
    markdown = "\n".join(element.value for element in at.markdown)
    subheaders = "\n".join(element.value for element in at.subheader)
    captions = "\n".join(element.value for element in at.caption)
    assert "Research Subtasks (up to 3)" in subheaders
    assert "planning is a separate phase" in captions.casefold()
    for task in tasks:
        assert task in markdown


def test_gap_labels_and_quotes_produce_one_actionable_query():
    malformed_model_output = (
        "Search query to fill the gap:\n"
        "\"Compare independent measurements of the device's output.\""
    )
    assert parse_gap_queries(malformed_model_output) == [
        "Compare independent measurements of the device's output."
    ]

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=malformed_model_output))]
    )
    with patch("agent.gap_detector.client.chat.completions.create", return_value=response):
        assert detect_gaps("Assess the device", ["Some evidence"]) == [
            "Compare independent measurements of the device's output."
        ]


def test_report_has_all_uncertainty_counts_and_no_partial_synthesis_table():
    source = {
        "source_id": "SRC-001",
        "title": "Primary source",
        "url": "https://example.com/source",
        "content": "The source passage exactly supports the claim.",
    }
    citations = [
        {
            "claim": f"Supported finding {index}",
            "status": "SUPPORTED",
            "verifier_status": "SUPPORTED",
            "source_id": "SRC-001",
            "origin_source_id": "SRC-001",
            "score": 0.9,
            "quote": "The source passage exactly supports the claim.",
        }
        for index in range(6)
    ] + [{
        "claim": "Unaligned evidence item",
        "status": "UNVERIFIED",
        "verifier_status": "SUPPORTED",
        "source_id": None,
        "origin_source_id": "SRC-001",
        "score": 0.05,
        "quote": "The source passage exactly supports the claim.",
    }]
    result = report_node({
        "question": "Research question",
        "evidence": ["Unaligned evidence item"],
        "claim_verifications": [],
        "sources": [source],
        "citations": citations,
        "iteration": MAX_GAP_ROUNDS,
        "gaps": ["Find an independent evaluation"],
    })
    report = result["report"]
    assert "- Supported: 6" in report
    assert "- Unverified: 1" in report
    assert f"Gap-research rounds: {MAX_GAP_ROUNDS} (max {MAX_GAP_ROUNDS})" in report
    assert "Find an independent evaluation" in report
    assert "| ---" not in report
    assert "> The source passage exactly supports the claim." in report


def test_report_counts_all_evidence_as_unverified_when_verifier_returns_none():
    evidence = [f"Candidate evidence {index}" for index in range(1, 9)]
    report = report_node({
        "question": "Research question",
        "evidence": evidence,
        "claim_verifications": [],
        "sources": [],
        "citations": [],
        "iteration": 0,
        "gaps": [],
    })["report"]
    assert "- Unverified: 8" in report
    assert all(f"UNVERIFIED** — {item}" in report for item in evidence)


def test_gap_round_warning_uses_configured_maximum():
    at = AppTest.from_file("app.py")
    at.session_state["result"] = {
        "report": "Report",
        "plan": [],
        "evidence": [],
        "citations": [],
        "sources": [],
        "gaps": ["Find an independent evaluation"],
        "iteration": MAX_GAP_ROUNDS,
    }
    at.run(timeout=60)
    warnings = "\n".join(element.value for element in at.warning)
    markdown = "\n".join(element.value for element in at.markdown)
    assert f"{MAX_GAP_ROUNDS}/{MAX_GAP_ROUNDS}" in warnings
    assert f"max rounds ({MAX_GAP_ROUNDS})" in markdown
    assert "<MAX_GAP_ROUNDS>" not in markdown


def test_rag_excludes_internal_overview_unless_query_is_about_this_project():
    store = VectorStore()
    store.vectors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        dtype="float32",
    )
    store.chunks = [
        {
            "name": "research_agent_overview.md",
            "chunk": 0,
            "text": "Internal architecture notes",
        },
        {
            "name": "framework_comparison.md",
            "chunk": 0,
            "text": "Framework comparison content",
        },
        {
            "name": "external_research.md",
            "chunk": 0,
            "text": "External subject matter research",
        },
    ]
    with (
        patch("RAG.retriever._get_store", return_value=store),
        patch("RAG.retriever.embed", return_value=np.array([[1.0, 0.0]], dtype="float32")),
    ):
        general_hits = kb_retrieve("Compare agent frameworks", k=3, min_score=0.0)
        project_hits = kb_retrieve("Explain this project's architecture", k=3, min_score=0.0)

    assert all(hit["title"] != "research_agent_overview.md" for hit in general_hits)
    assert any(hit["title"] == "research_agent_overview.md" for hit in project_hits)


def test_low_score_supported_assessment_renders_as_unverified_not_citation():
    at = AppTest.from_file("app.py")
    at.session_state["result"] = {
        "report": "Report",
        "plan": [],
        "evidence": ["A hotel API supports booking records."],
        "verified_claims": ["A hotel API supports booking records. [SUPPORTED]"],
        "citations": [{
            "claim": "A hotel API supports booking records. [SUPPORTED]",
            "status": "UNVERIFIED",
            "verifier_status": "SUPPORTED",
            "source_id": None,
            "origin_source_id": "SRC-001",
            "score": 0.05,
            "quote": "A dictionary lists words and their meanings.",
            "evidence_index": 0,
        }],
        "sources": [{
            "source_id": "SRC-001",
            "title": "Cambridge dictionary",
            "url": "https://dictionary.cambridge.org/",
            "content": "A dictionary lists words and their meanings.",
        }],
        "gaps": [],
        "iteration": 0,
    }
    at.run(timeout=60)

    assert not at.exception, at.exception
    markdown = "\n".join(element.value for element in at.markdown)
    captions = "\n".join(element.value for element in at.caption)
    expanders = "\n".join(element.label for element in at.expander)
    assert "**UNVERIFIED**" in markdown
    assert "0.05" in markdown
    assert "A dictionary lists words and their meanings." in markdown
    assert "Verifier assessment: SUPPORTED" in captions
    assert "View originating source (not cited)" in expanders
    report = report_node({
        "question": "Research question",
        "evidence": ["A hotel API supports booking records."],
        "claim_verifications": [],
        "citations": [at.session_state["result"]["citations"][0]],
        "sources": at.session_state["result"]["sources"],
        "iteration": 0,
        "gaps": [],
    })["report"]
    assert "- Supported: 0" in report
    assert "- Unverified: 1" in report
