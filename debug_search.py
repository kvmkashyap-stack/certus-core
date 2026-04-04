from tools.search_tool import search_web
import os

print("--- Testing Tavily Search ---")
results = search_web("What is Python 3.14?")

if not results:
    print("❌ ERROR: No results found. Check your TAVILY_API_KEY in .env")
else:
    print(f"✅ SUCCESS: Found {len(results)} results!")
    for r in results[:2]:
        print(f"- {r.get('title')}")