import os, requests, asyncio, uvicorn, faiss, shutil
import numpy as np
import pymupdf4llm
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

# --- AI TOOLS ---
def get_embeddings(texts):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    res = requests.post(EMBED_URL, headers=headers, json={"inputs": texts})
    # Basic error handling for API
    data = res.json()
    if isinstance(data, dict) and "error" in data:
        return [[0.0] * 384] # Fallback vector
    return data

def call_llm(prompt):
    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": "google/gemini-2.0-flash-001", "messages": [{"role": "user", "content": prompt}]}
        )
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling AI: {str(e)}"

def search_web(query):
    return [{"title": "Search Result", "url": "https://google.com", "content": f"Information about {query}..."}]

# --- VECTOR STORAGE ---
INDEX_FILE = "certus.index"
MAP_FILE = "certus.txt"

def process_file(path, name):
    ext = name.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf': text = pymupdf4llm.to_markdown(path)
        elif ext == 'docx': text = "\n".join([p.text for p in Document(path).paragraphs])
        
        if text:
            chunks = [text[i:i+500] for i in range(0, len(text), 400)]
            embeds = np.array(get_embeddings(chunks)).astype('float32')
            index = faiss.IndexFlatL2(embeds.shape[1])
            index.add(embeds)
            faiss.write_index(index, INDEX_FILE)
            with open(MAP_FILE, "a") as f:
                for c in chunks: f.write(c.replace('\n', ' ') + "\n")
        return f"Indexed {name}"
    except Exception as e:
        return f"Error processing {name}: {str(e)}"

# --- ENDPOINTS ---
@app.get("/")
async def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Certus Core is Live</h1><p>Upload index.html to see the full UI.</p>")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f: f.write(await file.read())
    msg = process_file(file.filename, file.filename)
    return {"message": msg}

@app.get("/api/research")
async def research(query: str = Query(...), mode: str = "General"):
    web_results = search_web(query)
    local_data = ""
    if os.path.exists(INDEX_FILE):
        q_vec = np.array(get_embeddings([query])).astype('float32')
        idx = faiss.read_index(INDEX_FILE)
        D, I = idx.search(q_vec, k=2)
        with open(MAP_FILE, "r") as f:
            lines = f.readlines()
            local_data = " ".join([lines[i] for i in I[0] if i < len(lines)])

    prompt = f"Context: {local_data}\nWeb: {web_results}\nQuestion: {query}\nAnswer professionally with citations."
    answer = call_llm(prompt)
    
    return {
        "answer": answer,
        "sources": [{"id": 1, "title": "Web Search", "url": "#"}]
    }

# --- RENDER PORT BINDING ---
if __name__ == "__main__":
    # Ensure port is an integer and defaults correctly for Render
    port = int(os.environ.get("PORT", 10000))
    # CRITICAL: Use the string "app:app" to help uvicorn locate the instance
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
