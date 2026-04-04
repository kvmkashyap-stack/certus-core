import os, uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tools.vector_tool import process_and_index, search_local
from tools.search_tool import web_research
from services.llm_service import ask_deepseek

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def home():
    return FileResponse("index.html") if os.path.exists("index.html") else HTMLResponse("Certus Core Live")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f: f.write(await file.read())
    msg = process_and_index(file.filename, file.filename)
    os.remove(file.filename)
    return {"message": msg}

@app.get("/api/research")
async def research(query: str = Query(...)):
    local_data = search_local(query)
    web_data = web_research(query)
    answer = ask_deepseek(query, local_data, web_data)
    return {"answer": answer, "sources": [{"id": 1, "title": "Certus Source", "url": "#"}]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)