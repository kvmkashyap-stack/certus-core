import requests
from config.settings import TAVILY_API_KEY

def web_research(query):
    """Uses Tavily for AI-optimized web search."""
    if not TAVILY_API_KEY:
        return [{"title": "Error", "content": "Tavily API Key missing.", "url": "#"}]
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json().get("results", [])
    except:
        return []