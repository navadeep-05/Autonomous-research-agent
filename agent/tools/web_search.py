from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query, max_results=3):
    """Search the web using Tavily. Returns list of {title, url, content}."""
    try:
        response = _client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception:
        return []