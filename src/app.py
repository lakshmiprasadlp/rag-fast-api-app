from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os

from rag_pipeline import process_document, ask_question

app = FastAPI()
os.makedirs("uploads", exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = process_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "File uploaded and indexed",
        "chunks_created": chunks
    }


@app.post("/query")
async def query(question: str):
    try:
        answer = ask_question(question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "question": question,
        "answer": answer
    }


@app.get("/health")
def health():
    return {"status": "OK"}
