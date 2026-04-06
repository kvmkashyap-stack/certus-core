import faiss
import numpy as np
import requests
import os
from config.settings import HF_TOKEN

# 384 is the standard dimension for all-MiniLM-L6-v2
DIMENSION = 384
INDEX_FILE = "certus.index"
TEXT_FILE = "certus.txt"

# We switch to IndexFlatIP (Inner Product). 
# When combined with normalized vectors, Inner Product = Cosine Similarity.
index = faiss.IndexFlatIP(DIMENSION)

def get_embeddings(texts):
    URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(URL, headers=headers, json={"inputs": texts})
    vectors = np.array(response.json()).astype('float32')
    
    # --- STEP 1: NORMALIZE FOR COSINE SIMILARITY ---
    faiss.normalize_L2(vectors) 
    return vectors

def process_and_store_file(file_path, filename):
    # (Extraction logic remains the same: PDF/Docx to text)
    # ... [Assuming text extraction happens here] ...
    raw_text = "Extracted text from your file..." 
    
    new_vectors = get_embeddings([raw_text])
    
    # --- STEP 2: ADD TO INNER PRODUCT INDEX ---
    index.add(new_vectors)
    faiss.write_index(index, INDEX_FILE)
    
    with open(TEXT_FILE, "a") as f:
        f.write(f"SOURCE:{filename}|CONTENT:{raw_text}\n")
    
    return f"Indexed {filename} with Cosine Similarity."

def search_local(query, k=3):
    if not os.path.exists(INDEX_FILE): return ""
    
    # Load index and encode query
    current_index = faiss.read_index(INDEX_FILE)
    query_vector = get_embeddings([query])
    
    # --- STEP 3: SEARCH ---
    # D = Similarity Scores (Closer to 1.0 is a perfect match)
    # I = Indices
    D, I = current_index.search(query_vector, k)
    
    # ... [Logic to pull text from certus.txt based on I] ...
    return "Relevant context found via Cosine Similarity..."
