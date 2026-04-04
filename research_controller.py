from fastapi import APIRouter, UploadFile, File
from tools.vector_tool import process_and_store_file
import shutil
import os

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = process_and_store_file(temp_path, file.filename)
    os.remove(temp_path)
    return {"message": result}

@router.get("/research")
async def research(query: str):
    from services.research_service import handle_query
    return await handle_query(query)