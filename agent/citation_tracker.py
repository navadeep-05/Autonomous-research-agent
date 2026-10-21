def create_source_records(search_results):
    """
    Create unique IDs for every research source.
    """

    sources = []
    counter = 1

    for item in search_results:
        result = item["result"]
        # result is now a list of {title, url, content} dicts
        entries = result if isinstance(result, list) else [{"url": "", "title": "", "content": str(result)}]
        for entry in entries:
            sources.append({
                "source_id": f"SRC-{counter:03d}",
                "task": item["task"],
                "url": entry.get("url", ""),
                "title": entry.get("title", ""),
                "content": entry.get("content", ""),
            })
            counter += 1

    return sources


def create_citation(claim, source_id):
    """
    Create a simple citation reference.
    """

    return {
        "claim": claim,
        "source_id": source_id
    }