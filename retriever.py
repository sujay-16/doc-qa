import os
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_relevant_chunks(question: str, collection_name: str, n_results: int = 4) -> list[str]:
    """
    Embed the question and search ChromaDB for the most relevant chunks.
    Returns a list of text chunks ordered by relevance.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
    )

    return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    # Quick test — replace with your actual collection name
    collection = input("Enter collection name: ")
    question = input("Enter a test question: ")
    chunks = get_relevant_chunks(question, collection)
    print(f"\nTop {len(chunks)} relevant chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk[:200])
        print()