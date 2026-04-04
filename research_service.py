import json
from tools.vector_tool import search_local
from tools.search_tool import web_research
from tools.llm_tool import run_llm_call

HISTORY_FILE = "archive.json"

async def handle_query(query: str, mode: str = "General"):
    # 1. Hybrid Search
    local_context = search_local(query)
    web_results = web_research(query)
    web_text = "\n".join([f"{r['title']}: {r['content']}" for r in web_results])
    
    # 2. Deep Thinking Prompt
    prompt = f"""
    You are Certus Core (Research Mode: {mode}).
    CONTEXT FROM DOCUMENTS: {local_context}
    CONTEXT FROM WEB: {web_text}
    
    QUERY: {query}
    
    INSTRUCTIONS: 
    1. Think step-by-step.
    2. Cite sources using [1], [2].
    3. If data conflicts, prioritize recent web data.
    """
    
    # 3. Call DeepSeek-R1 (OpenRouter handles reasoning tokens)
    answer = run_llm_call(prompt, model="deepseek/deepseek-r1")
    
    # 4. Auto-Archive
    record = {"query": query, "answer": answer, "mode": mode}
    save_to_archive(record)
    
    return {
        "answer": answer,
        "sources": web_results,
        "thinking": "Analysis complete using DeepSeek-R1 reasoning engine."
    }

def save_to_archive(data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f: history = json.load(f)
    history.append(data)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f)