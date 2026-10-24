import re
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from agent.state import create_initial_state
from agent.planner import create_research_plan
from agent.tools.web_search import search_web
from agent.evidence import extract_evidence
from agent.source_evaluator import evaluate_source
from agent.gap_detector import detect_gaps
from agent.citation_tracker import create_source_records, create_citation
from agent.claim_verifier import parse_verification_results, verify_claims
from agent.report_generator import generate_report
from RAG.embeddings import embed
from RAG.retriever import retrieve as kb_retrieve

MAX_TASKS = 3
MAX_GAP_ROUNDS = 1
MAX_GAPS_PER_ROUND = 1
LOCAL_K = 2
MIN_CITE_SCORE = 0.30


class ResearchState(TypedDict):
    question: str
    plan: list
    search_results: list
    evidence: list
    evidence_origins: list
    source_evaluations: list
    gaps: list
    verified_claims: list
    claim_verifications: list
    sources: list
    citations: list
    report: Any
    current_task: int
    iteration: int


# ---------------- helpers ----------------
def source_kind(url):
    low = (url or "").lower()
    if low.startswith("local://"):
        return "Local document (RAG)"
    if low.endswith(".pdf") or "arxiv.org" in low or "doi.org" in low:
        return "Research paper / PDF"
    return "Web"


def claim_status(text):
    up = text.upper()
    match = re.search(
        r"\[(PARTIALLY SUPPORTED|PARTIAL|UNSUPPORTED|SUPPORTED|UNKNOWN|UNVERIFIED)\]",
        up,
    )
    if match:
        return {
            "PARTIALLY SUPPORTED": "PARTIAL",
            "UNKNOWN": "UNKNOWN",
            "UNVERIFIED": "UNKNOWN",
        }.get(match.group(1), match.group(1))
    return "UNKNOWN"


def parse_claims(text):
    if not text:
        return []
    items = []
    for line in str(text).splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Match numbered lists (1., 1), etc.), markdown bold numbered items (**1.**), or bullet points with status tags
        if re.match(r"^(\W*\d+[.)]|\*\*+\d+[.)]\*\*+|[-*•])\s*", cleaned):
            items.append(cleaned)
        elif items and not cleaned.startswith(("#", "```", "---")):
            items[-1] += " " + cleaned
    if not items and str(text).strip():
        items = [line.strip() for line in str(text).splitlines() if line.strip() and not line.strip().startswith(("#", "```"))]
    return items


def format_sources(sources):
    lines = ["\n\n---\n\n## Sources\n"]
    for s in sources:
        url = s.get("url", "")
        title = s.get("title") or url or "Untitled"
        if url.startswith("local://"):
            lines.append(f"- **{s['source_id']}** {title} — Local document (RAG) — `{url}`")
        else:
            lines.append(f"- **{s['source_id']}** [{title}]({url}) — {source_kind(url)}")
    return "\n".join(lines)


def research_one(query):
    """Web search + local RAG -> evidence + source evaluation."""
    query = str(query)
    web = search_web(query) or []
    local = kb_retrieve(query, k=LOCAL_K)
    combined = list(web) + list(local)
    if not combined:
        return ({"task": query, "result": []}, [], [],
                {"task": query, "evaluation": "No sources found."})

    new_evidence, evidence_origins = [], []
    for source in combined:
        extracted = extract_evidence(query, [source])
        if isinstance(extracted, str):
            extracted = [extracted]
        new_evidence.extend(extracted)
        evidence_origins.extend([source.get("url", "")] * len(extracted))

    text = "\n\n".join(f"[{r.get('url', '')}]\n{r.get('content', '')[:500]}" for r in combined[:4])
    evaluation = evaluate_source(text)
    return ({"task": query, "result": combined}, list(new_evidence), evidence_origins,
            {"task": query, "evaluation": evaluation})


def _run_queries(state, queries):
    results = list(state["search_results"])
    evidence = list(state["evidence"])
    evidence_origins = list(state.get("evidence_origins", []))
    evals = list(state["source_evaluations"])
    for q in queries:
        entry, new_ev, new_origins, ev = research_one(q)
        results.append(entry)
        evidence.extend(new_ev)
        evidence_origins.extend(new_origins)
        evals.append(ev)
    return {
        "search_results": results,
        "evidence": evidence,
        "evidence_origins": evidence_origins,
        "source_evaluations": evals,
    }


# ---------------- nodes ----------------
def plan_node(state):
    return {"plan": list(create_research_plan(state["question"]))[:MAX_TASKS]}


def research_node(state):
    out = _run_queries(state, state["plan"])
    out["current_task"] = max(len(state["plan"]) - 1, 0)
    return out


def gap_detect_node(state):
    return {"gaps": list(detect_gaps(state["question"], state["evidence"]))}


def gap_research_node(state):
    out = _run_queries(state, state["gaps"][:MAX_GAPS_PER_ROUND])
    out["iteration"] = state["iteration"] + 1
    return out


def sources_node(state):
    seen, unique = set(), []
    for s in create_source_records(state["search_results"]):
        key = s["url"] or s["source_id"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return {"sources": unique}


def verify_node(state):
    if not state["evidence"]:
        return {
            "verified_claims": [],
            "claim_verifications": [],
            "verification_error": "",
        }
    try:
        sources = state.get("sources", [])
        source_id_by_url = {
            source.get("url"): source.get("source_id")
            for source in sources
            if source.get("url") and source.get("source_id")
        }
        origins = state.get("evidence_origins", [])
        allowed_source_ids = [
            [source_id_by_url[origins[index]]]
            if index < len(origins) and origins[index] in source_id_by_url
            else []
            for index in range(len(state["evidence"]))
        ]
        results = parse_verification_results(
            verify_claims(
                state["evidence"],
                sources,
                allowed_source_ids=allowed_source_ids,
                raise_on_error=True,
            ),
            state["evidence"],
            sources,
            allowed_source_ids=allowed_source_ids,
        )
    except Exception as exc:
        print(f"[graph] Claim verification failed: {exc}")
        return {
            "verified_claims": [],
            "claim_verifications": [],
            "verification_error": "Claim verification failed. Check the server logs and verifier configuration.",
        }
    claims = [f"{result['claim']} [{result['status']}]" for result in results]
    verification_warnings = [
        result["verification_warning"]
        for result in results
        if result.get("verification_warning")
    ]
    return {
        "verified_claims": claims,
        "claim_verifications": results,
        "verification_error": "; ".join(verification_warnings),
    }


def citations_node(state):
    verifications = state.get("claim_verifications", [])
    sources = state["sources"]
    evidence = state["evidence"]
    evidence_origins = state.get("evidence_origins", [])
    if not sources:
        return {
            "citations": [
                {
                    "claim": result["claim"],
                    "status": (
                        "UNVERIFIED"
                        if result["status"] in {"SUPPORTED", "PARTIAL"}
                        else result["status"]
                    ),
                    "verifier_status": result["status"],
                    "source_id": None,
                    "score": None,
                    "quote": "",
                    "evidence_index": result["evidence_index"],
                    "origin_source_id": (
                        result["allowed_source_ids"][0]
                        if len(result.get("allowed_source_ids", [])) == 1
                        else None
                    ),
                }
                for result in verifications
            ],
            "evidence_matches": [
                {
                    "evidence": item,
                    "source_id": None,
                    "origin_url": evidence_origins[index]
                    if index < len(evidence_origins) else None,
                    "score": None,
                }
                for index, item in enumerate(evidence)
            ],
        }
    src_vecs = embed([(s.get("content") or s.get("title") or "")[:1000] for s in sources])
    source_indexes_by_id = {
        source.get("source_id"): index for index, source in enumerate(sources)
    }
    source_id_by_url = {
        source.get("url"): source.get("source_id")
        for source in sources
        if source.get("url") and source.get("source_id")
    }
    cites = []
    claim_sims = embed([verification["claim"] for verification in verifications]) @ src_vecs.T if verifications else []
    for i, verification in enumerate(verifications):
        claim = verification["claim"]
        match = claim_sims[i]
        allowed_ids = verification.get("allowed_source_ids", [])
        allowed_indexes = [
            source_indexes_by_id[source_id]
            for source_id in allowed_ids
            if source_id in source_indexes_by_id
        ]
        selected_id = verification.get("source_id")
        selected_index = next(
            (
                index for index, source in enumerate(sources)
                if source.get("source_id") == selected_id and index in allowed_indexes
            ),
            None,
        )
        j = selected_index
        if j is None and allowed_indexes:
            j = max(allowed_indexes, key=lambda index: float(match[index]))
        score = float(match[j]) if j is not None else None
        source_id = (
            sources[j]["source_id"]
            if j is not None and score is not None and score >= MIN_CITE_SCORE
            else None
        )
        verifier_status = verification["status"]
        status = verifier_status
        if status in {"SUPPORTED", "PARTIAL"} and source_id is None:
            status = "UNVERIFIED"
        cite = create_citation(claim, source_id)
        cite["status"] = status
        cite["verifier_status"] = verifier_status
        cite["quote"] = verification.get("quote", "")
        cite["evidence_index"] = verification["evidence_index"]
        cite["origin_source_id"] = (
            allowed_ids[0] if len(allowed_ids) == 1 else None
        )
        cite["score"] = round(score, 2) if score is not None else None
        cites.append(cite)

    evidence_matches = []
    if evidence:
        evidence_sims = embed([str(item) for item in evidence]) @ src_vecs.T
        for i, item in enumerate(evidence):
            origin_url = evidence_origins[i] if i < len(evidence_origins) else None
            origin_id = source_id_by_url.get(origin_url)
            j = source_indexes_by_id.get(origin_id)
            score = float(evidence_sims[i][j]) if j is not None else None
            evidence_matches.append({
                "evidence": item,
                "source_id": origin_id,
                "origin_url": origin_url,
                "score": round(score, 2) if score is not None else None,
            })
    return {"citations": cites, "evidence_matches": evidence_matches}


def report_node(state):
    body = generate_report(
        state["question"],
        state["evidence"],
        state.get("claim_verifications", []),
        state["sources"],
        state["citations"],
        MIN_CITE_SCORE,
    )
    status_counts = {
        status: sum(
            1 for cite in state["citations"]
            if (cite.get("status") or claim_status(cite.get("claim", ""))) == status
        )
        for status in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNKNOWN", "UNVERIFIED")
    }
    status_counts["UNVERIFIED"] += max(
        0,
        len(state["evidence"]) - len(state["citations"]),
    )
    footer = [
        "\n\n## Verification and completeness",
        f"- Claims checked: {len(state['citations'])}",
        f"- Supported: {status_counts['SUPPORTED']}",
        f"- Partially supported: {status_counts['PARTIAL']}",
        f"- Unsupported: {status_counts['UNSUPPORTED']}",
        f"- Unknown: {status_counts['UNKNOWN']}",
        f"- Unverified: {status_counts['UNVERIFIED']}",
        f"- Gap-research rounds: {state['iteration']} (max {MAX_GAP_ROUNDS})",
    ]
    footer.append(
        "- Unresolved gaps:\n" + "\n".join(f"  - {gap}" for gap in state["gaps"])
        if state["gaps"] else "- Unresolved gaps: none"
    )
    return {"report": body + "\n\n" + "\n".join(footer) + format_sources(state["sources"])}


def route_after_gaps(state):
    if state["gaps"] and state["iteration"] < MAX_GAP_ROUNDS:
        return "gap_researcher"
    return "source_recorder"


# ---------------- graph ----------------
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", plan_node)
    g.add_node("researcher", research_node)
    g.add_node("gap_detector", gap_detect_node)
    g.add_node("gap_researcher", gap_research_node)
    g.add_node("source_recorder", sources_node)
    g.add_node("verifier", verify_node)
    g.add_node("citation_tracker", citations_node)
    g.add_node("reporter", report_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "gap_detector")
    g.add_conditional_edges("gap_detector", route_after_gaps,
                            {"gap_researcher": "gap_researcher",
                             "source_recorder": "source_recorder"})
    g.add_edge("gap_researcher", "gap_detector")
    g.add_edge("source_recorder", "verifier")
    g.add_edge("verifier", "citation_tracker")
    g.add_edge("citation_tracker", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


def stream_graph(question):
    """Yield (node_name, state_so_far) after every node finishes."""
    compiled = build_graph()
    state = create_initial_state(question)
    for update in compiled.stream(dict(state), stream_mode="updates"):
        for node, delta in update.items():
            state.update(delta or {})
            yield node, state


def run_graph(question):
    final = None
    for _, final in stream_graph(question):
        pass
    return final