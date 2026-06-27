import os
import hashlib
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    return OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )


def load_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for better context preservation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def get_collection_name(pdf_path: str) -> str:
    """Generate a unique collection name based on the filename."""
    filename = os.path.basename(pdf_path)
    hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
    safe_name = "".join(c if c.isalnum() else "_" for c in filename[:20])
    return f"{safe_name}_{hash_suffix}"


def embed_and_store(chunks: list[str], collection_name: str) -> int:
    """Embed chunks using Ollama and store them in ChromaDB."""
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    # Delete existing collection if it exists (for re-ingestion)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    # Add chunks in batches
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    return len(chunks)


def ingest_pdf(pdf_path: str) -> dict:
    """Full pipeline: load PDF → chunk → embed → store. Returns collection info."""
    print(f"Loading PDF: {pdf_path}")
    text = load_pdf(pdf_path)

    if not text.strip():
        raise ValueError("Could not extract text from PDF. It may be scanned/image-based.")

    print(f"Chunking text ({len(text)} characters)...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    collection_name = get_collection_name(pdf_path)
    print(f"Embedding and storing in ChromaDB (collection: {collection_name})...")
    count = embed_and_store(chunks, collection_name)
    print(f"Done! Stored {count} chunks.")

    return {
        "collection_name": collection_name,
        "chunk_count": count,
        "filename": os.path.basename(pdf_path),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestor.py <path_to_pdf>")
        sys.exit(1)
    result = ingest_pdf(sys.argv[1])
    print(result)