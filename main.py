# Now do your imports
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, shutil
from services.research_service import handle_query
from tools.vector_tool import process_and_store_file
from fastapi.responses import FileResponse


import os, uvicorn, shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.research_service import handle_query
from tools.vector_tool import process_and_store_file
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def serve_ui(): return FileResponse("index.html")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    temp = f"temp_{file.filename}"
    with open(temp, "wb") as b: shutil.copyfileobj(file.file, b)
    msg = process_and_store_file(temp, file.filename)
    os.remove(temp)
    return {"message": msg}

@app.get("/research")
async def research(query: str, mode: str = "General"):
    return await handle_query(query, mode)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)