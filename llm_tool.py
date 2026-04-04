import requests
from config.settings import OPENROUTER_API_KEY

def generate_answer(query: str, context: str):
    """The Architect: Uses DeepSeek-R1 for deep reasoning."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nProvide a detailed, expert research report with citations."
    # We use DeepSeek-R1 here for the 'Heavy Lifting'
    return run_llm_call(prompt, model="deepseek/deepseek-r1")

def fact_check_answer(draft: str, context: str):
    """The Auditor: Uses Gemini Flash for speed."""
    prompt = f"DRAFT: {draft}\nSOURCES: {context}\nFix hallucinations and ensure citations are correct."
    return run_llm_call(prompt, model="google/gemini-2.0-flash-001")

def run_llm_call(prompt, model="google/gemini-2.0-flash-001"):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return r.json()['choices'][0]['message']['content']