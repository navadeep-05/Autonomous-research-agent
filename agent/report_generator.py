def generate_report(
    question,
    evidence,
    verified_claims,
    sources,
    citations=None,
    min_cite_score=0.30,
):
    """Render source-linked assessment results without synthesizing new claims."""
    citations = list(citations or [])
    sources_by_id = {
        source.get("source_id"): source
        for source in sources
        if source.get("source_id")
    }
    lines = [
        "# Research report",
        "",
        f"**Research question:** {question}",
        "",
        "This report lists evidence assessments from the supplied source passages. "
        "Semantic similarity is used for source alignment, not as proof of factual truth.",
        "",
        "## Evidence assessments",
        "",
    ]

    if citations:
        covered_indexes = {
            citation.get("evidence_index")
            for citation in citations
            if isinstance(citation.get("evidence_index"), int)
        }
        for citation in citations:
            status = citation.get("status", "UNVERIFIED")
            quote = citation.get("quote", "")
            source_id = citation.get("source_id")
            source = sources_by_id.get(source_id)
            origin_source = sources_by_id.get(citation.get("origin_source_id"))
            score = citation.get("score")
            score_passed = isinstance(score, (int, float)) and score >= min_cite_score
            if status in {"SUPPORTED", "PARTIAL"} and (
                source is None
                or not quote
                or quote not in source.get("content", "")
                or not score_passed
            ):
                status = "UNVERIFIED"

            lines.extend([
                f"### {status}: {citation.get('claim', '')}",
                "",
            ])
            verifier_status = citation.get("verifier_status")
            if verifier_status and verifier_status != status:
                lines.append(f"- Verifier assessment before alignment check: {verifier_status}")
            if source:
                lines.append(f"- Source: [{source_id}]({source.get('url') or source.get('title', '')})")
                lines.append(f"- Cosine similarity: {score:.2f}")
            elif origin_source:
                lines.append(
                    f"- Origin source (not cited; alignment cutoff not met): "
                    f"{origin_source.get('title') or origin_source.get('url') or citation.get('origin_source_id')}"
                )
            else:
                lines.append("- Source: no aligned source citation")
            if quote and (source or origin_source):
                lines.extend(["", f"> {quote}"])
            lines.append("")
        unrepresented = [
            item for index, item in enumerate(evidence)
            if index not in covered_indexes
        ]
        if unrepresented:
            lines.extend([
                "### UNVERIFIED evidence without a verifier result",
                "",
                *[f"- {item}" for item in unrepresented],
                "",
            ])
    elif evidence:
        lines.extend([
            "No claims could be verified against their originating sources. "
            "The extracted statements below are unverified candidates, not confirmed facts.",
            "",
        ])
        lines.extend(f"- **UNVERIFIED** — {item}" for item in evidence)
        lines.append("")
    else:
        lines.extend(["No evidence was extracted.", ""])

    return "\n".join(lines).rstrip()
