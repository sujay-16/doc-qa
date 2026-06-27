import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingestor import ingest_pdf
from qa_chain import answer_question
from fastapi.responses import StreamingResponse
from retriever import get_relevant_chunks
import ollama

app = FastAPI(
    title="Document Q&A API",
    description="Upload a PDF and ask questions about it using a local LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    collection_name: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    collection_name: str


@app.get("/")
def root():
    return {"message": "Document Q&A API is running. Visit /docs for the API explorer."}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and ingest it into the vector database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path)
        result["filename"] = file.filename
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Ask a question about an uploaded document."""
    try:
        result = answer_question(request.question, request.collection_name)
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            collection_name=request.collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    """List all ingested document collections."""
    import chromadb
    client = chromadb.PersistentClient(path="chroma_db")
    collections = client.list_collections()
    return {"collections": [c.name for c in collections]}