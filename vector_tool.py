import os, requests, faiss
import numpy as np
import pymupdf4llm
from docx import Document
from config.settings import HF_TOKEN

EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "certus.index"
MAP_FILE = "certus.txt"

def get_embeddings(texts):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        res = requests.post(EMBED_URL, headers=headers, json={"inputs": texts}, timeout=10)
        return res.json()
    except:
        return [[0.0] * 384]

def process_and_store_file(path, name): # Matched to controller name
    ext = name.split('.')[-1].lower()
    text = ""
    if ext == 'pdf': text = pymupdf4llm.to_markdown(path)
    elif ext == 'docx': text = "\n".join([p.text for p in Document(path).paragraphs])
    
    if text:
        chunks = [text[i:i+600] for i in range(0, len(text), 500)]
        embeds = np.array(get_embeddings(chunks)).astype('float32')
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)
        faiss.write_index(index, INDEX_FILE)
        with open(MAP_FILE, "a") as f:
            for c in chunks: f.write(c.replace('\n', ' ') + "\n")
    return f"Successfully Indexed {name}"

def search_local(query):
    if not os.path.exists(INDEX_FILE): return ""
    q_vec = np.array(get_embeddings([query])).astype('float32')
    idx = faiss.read_index(INDEX_FILE)
    D, I = idx.search(q_vec, k=3)
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, "r") as f:
            lines = f.readlines()
            return " ".join([lines[i] for i in I[0] if i < len(lines)])
    return ""