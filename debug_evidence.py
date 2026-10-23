"""Debug evidence extraction to see why it returns empty."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from agent.tools.web_search import search_web
from agent.evidence import extract_evidence

results = search_web("What is quantum computing?")
print(f"Web results: {len(results)}")
for r in results:
    title = r.get("title", "")[:60]
    content_len = len(r.get("content", ""))
    print(f"  title: {title}, content len: {content_len}")

evidence = extract_evidence("What is quantum computing?", results)
print(f"\nEvidence extracted: {len(evidence)}")
for e in evidence[:5]:
    print(f"  > {str(e)[:120]}")
