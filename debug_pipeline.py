"""Debug script to trace the full pipeline and identify where claims/citations fail."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from agent.graph import stream_graph, parse_claims

for node, state in stream_graph("What is quantum computing?"):
    print(f"NODE: {node}")
    if node == "researcher":
        ev = state.get("evidence", [])
        sr = state.get("search_results", [])
        print(f"  evidence count: {len(ev)}")
        print(f"  search_results count: {len(sr)}")
    if node == "gap_detector":
        gaps = state.get("gaps", [])
        print(f"  gaps: {gaps}")
    if node == "verifier":
        vc = state.get("verified_claims", [])
        print(f"  verified_claims count: {len(vc)}")
        for c in vc[:5]:
            print(f"  > {str(c)[:120]}")
    if node == "citation_tracker":
        ci = state.get("citations", [])
        print(f"  citations count: {len(ci)}")
        for c in ci[:3]:
            print(f"  > claim={str(c.get('claim',''))[:80]}  src={c.get('source_id')}  score={c.get('score')}")
    if node == "reporter":
        rpt = str(state.get("report", ""))
        print(f"  report length: {len(rpt)}")
        print(f"  report preview: {rpt[:200]}")

print("\nDONE")
