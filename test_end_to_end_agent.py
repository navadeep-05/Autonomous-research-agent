"""
End-to-End Autonomous Research Agent Validation Suite
Tests:
1. Local Knowledge Base Indexing (SentenceTransformers + NumPy)
2. Dual Retrieval (Web + Local RAG)
3. Full 10-step LangGraph Workflow Execution
4. Gap Detection & Bounded Re-Search Loop (<= 2 rounds)
5. Claim Verification & Semantic Vector Citation Matching (cosine >= 0.30)
6. Final Cited Report Generation & Source Ledger
"""

import os
import sys

# Force UTF-8 stdout for Windows consoles
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, ".")

from RAG.retriever import build_index, index_size, retrieve as kb_retrieve
from agent.tools.web_search import search_web
from agent.graph import run_graph, stream_graph, claim_status


def test_1_rag_indexing():
    print("\n--- TEST 1: Local Knowledge Base Indexing ---")
    indexed_count = build_index()
    print(f"Total Chunks Indexed: {indexed_count} (Store size: {index_size()})")
    assert indexed_count > 0, "RAG indexing failed: 0 chunks created."
    
    hits = kb_retrieve("Majorana zero modes topological qubits", k=2, min_score=0.15)
    print(f"Retrieved Hits for 'Majorana zero modes': {len(hits)}")
    for h in hits:
        print(f"  • [{h['score']:.2f}] {h['title']} -> {h['url']}")
    assert len(hits) > 0, "RAG retrieval returned no hits for local document query."
    print("✓ TEST 1 PASSED: Local RAG storage and dense retrieval verified.")


def test_2_web_search():
    print("\n--- TEST 2: Web Search Live Tool ---")
    results = search_web("Topological quantum computing Majorana", max_results=2)
    print(f"Web Results Retrieved: {len(results)}")
    for r in results:
        print(f"  • {r['title']} ({r['url']})")
    assert len(results) > 0, "Web search returned 0 results. Check TAVILY_API_KEY."
    print("✓ TEST 2 PASSED: Tavily live web retrieval verified.")


def test_3_end_to_end_graph_execution():
    print("\n--- TEST 3: Full 10-Step LangGraph Workflow ---")
    question = "How do topological qubits achieve fault tolerance using Majorana zero modes?"
    print(f"Testing Question: '{question}'")
    
    nodes_executed = []
    final_state = None
    
    for node, current_state in stream_graph(question):
        nodes_executed.append(node)
        print(f"  -> Finished Node: {node}")
        final_state = current_state

    print(f"\nExecution Path: {' -> '.join(nodes_executed)}")
    
    # 1. Check Planner
    plan = final_state.get("plan", [])
    print(f"1. Plan Tasks ({len(plan)}): {plan}")
    assert len(plan) > 0, "No research tasks were planned."
    
    # 2. Check Evidence
    evidence = final_state.get("evidence", [])
    print(f"2. Evidence Extracted ({len(evidence)} facts)")
    assert len(evidence) > 0, "No evidence facts were extracted."
    
    # 3. Check Sources (Dual: Web + Local RAG)
    sources = final_state.get("sources", [])
    n_local = sum(1 for s in sources if s.get("url", "").startswith("local://"))
    n_web = len(sources) - n_local
    print(f"3. Sources Ledger: Total={len(sources)} (Web={n_web}, Local RAG={n_local})")
    assert len(sources) > 0, "No sources were recorded."
    assert n_local > 0, "Expected at least 1 local RAG source from data/."
    
    # 4. Check Gap Loop
    iterations = final_state.get("iteration", 0)
    gaps = final_state.get("gaps", [])
    print(f"4. Re-Search Gap Loop: Iterations={iterations} (Max=2), Remaining Gaps={len(gaps)}")
    assert iterations <= 2, f"Iterations exceeded maximum allowed limit of 2: {iterations}"
    
    # 5. Check Claim Verification & Citations
    citations = final_state.get("citations", [])
    print(f"5. Audited Claims & Citations: {len(citations)}")
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0, "UNKNOWN": 0}
    for c in citations:
        st = claim_status(c.get("claim", ""))
        counts[st] = counts.get(st, 0) + 1
        print(f"   [{st}] {c['claim'][:80]}... -> Ref: {c.get('source_id')} (Sim: {c.get('score')})")
    print(f"   Status Breakdown: {counts}")
    assert len(citations) > 0, "No claims were verified or cited."
    
    # 6. Check Final Report
    report = final_state.get("report", "")
    if isinstance(report, list):
        report = "\n".join(str(x) for x in report)
    print(f"6. Final Report Length: {len(report)} characters")
    assert len(report) > 300, "Report generation produced an unexpectedly short response."
    assert "[SRC-" in report or "## Sources" in report, "Report lacks source references."
    print("✓ TEST 3 PASSED: End-to-end LangGraph coordination verified.")


if __name__ == "__main__":
    test_1_rag_indexing()
    test_2_web_search()
    test_3_end_to_end_graph_execution()
    print("\n==================================================")
    print("🎉 ALL END-TO-END VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
