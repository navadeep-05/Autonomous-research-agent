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
from agent.claim_verifier import verify_claims
from agent.report_generator import generate_report
from RAG.embeddings import embed
from RAG.retriever import retrieve as kb_retrieve

MAX_TASKS = 2
MAX_GAP_ROUNDS = 1
MAX_GAPS_PER_ROUND = 1
LOCAL_K = 2
MIN_CITE_SCORE = 0.30


class ResearchState(TypedDict):
    question: str
    plan: list
    search_results: list
    evidence: list
    source_evaluations: list
    gaps: list
    verified_claims: list
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
    for part in (up[-30:], up):
        if "UNSUPPORTED" in part:
            return "UNSUPPORTED"
        if "PARTIALLY" in part:
            return "PARTIAL"
        if "SUPPORTED" in part:
            return "SUPPORTED"
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
        return ({"task": query, "result": []}, [],
                {"task": query, "evaluation": "No sources found."})
    new_evidence = extract_evidence(query, combined)
    if isinstance(new_evidence, str):
        new_evidence = [new_evidence]
    text = "\n\n".join(f"[{r.get('url', '')}]\n{r.get('content', '')[:500]}" for r in combined[:4])
    evaluation = evaluate_source(text)
    return ({"task": query, "result": combined}, list(new_evidence),
            {"task": query, "evaluation": evaluation})


def _run_queries(state, queries):
    results = list(state["search_results"])
    evidence = list(state["evidence"])
    evals = list(state["source_evaluations"])
    for q in queries:
        entry, new_ev, ev = research_one(q)
        results.append(entry)
        evidence.extend(new_ev)
        evals.append(ev)
    return {"search_results": results, "evidence": evidence, "source_evaluations": evals}


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
        return {"verified_claims": []}
    return {"verified_claims": parse_claims(verify_claims(state["evidence"]))}


def citations_node(state):
    claims, sources = state["verified_claims"], state["sources"]
    if not claims or not sources:
        return {"citations": []}
    src_vecs = embed([(s.get("content") or s.get("title") or "")[:1000] for s in sources])
    cleaned = [re.sub(r"\[(SUPPORTED|PARTIALLY SUPPORTED|PARTIAL|UNSUPPORTED)\]", "", re.sub(r"^\W*\d+[.)]\s*", "", c), flags=re.I).strip() for c in claims]
    sims = embed(cleaned) @ src_vecs.T
    cites = []
    for i, claim in enumerate(claims):
        j = int(np.argmax(sims[i]))
        score = float(sims[i][j])
        cite = create_citation(claim, sources[j]["source_id"] if score >= MIN_CITE_SCORE else None)
        cite["score"] = round(score, 2)
        cites.append(cite)
    return {"citations": cites}


def report_node(state):
    claim_lines = [
        f"{c['claim']} [{c['source_id']}]" if c.get("source_id") else f"{c['claim']} [no source match]"
        for c in state["citations"]
    ]
    body = generate_report(state["question"], state["evidence"],
                           "\n".join(claim_lines), state["sources"])
    if isinstance(body, list):
        body = "\n\n".join(str(x) for x in body)

    counts = {}
    for c in state["citations"]:
        k = claim_status(c["claim"])
        counts[k] = counts.get(k, 0) + 1
    footer = [
        "\n\n---\n\n## Verification & Completeness",
        f"- Claims checked: {len(state['citations'])} (supported {counts.get('SUPPORTED', 0)}, "
        f"partially supported {counts.get('PARTIAL', 0)}, unsupported {counts.get('UNSUPPORTED', 0)})",
        f"- Gap-research rounds: {state['iteration']} (max {MAX_GAP_ROUNDS})",
    ]
    if state["gaps"]:
        footer.append("- Unresolved gaps:")
        footer += [f"  - {g}" for g in state["gaps"]]
    else:
        footer.append("- Unresolved gaps: none")

    return {"report": str(body) + "\n".join(footer) + format_sources(state["sources"])}


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