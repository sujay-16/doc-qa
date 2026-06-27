# Architecture Decisions

This document explains the key technical choices made in building this project and the reasoning behind each one.

---

## 1. Why RAG instead of fine-tuning?

**Fine-tuning** means retraining a model on your specific data. It's expensive (requires GPUs and hours of compute), and the model's knowledge becomes static — if the document changes, you have to retrain.

**RAG (Retrieval-Augmented Generation)** keeps the LLM frozen and instead retrieves relevant context at query time. This means:
- Works on any document instantly — no retraining
- Answers stay grounded in the actual document (less hallucination)
- Can handle new documents without any model changes
- Much cheaper — no GPU training required

For a document Q&A use case where documents change frequently, RAG is the right choice over fine-tuning.

---

## 2. Why Ollama instead of the OpenAI API?

**OpenAI API** costs money per token and requires sending your documents to a third-party server. For sensitive documents (legal, medical, financial), this is a privacy concern.

**Ollama** runs the LLM locally on your machine:
- Zero cost — no API bills regardless of usage
- Complete privacy — documents never leave your machine
- No rate limits or downtime dependency
- Works offline

The tradeoff is that local models (llama3 7B) are less capable than GPT-4, but for document Q&A with focused context, the quality is more than sufficient.

---

## 3. Why ChromaDB instead of Pinecone or Weaviate?

**Pinecone and Weaviate** are cloud vector databases. They require API keys, have free tier limits, and add network latency to every query.

**ChromaDB** runs locally and persists to disk:
- Zero configuration — works out of the box
- No signup or API key needed
- Persistent across sessions (stored in `chroma_db/` folder)
- Fast enough for personal/small-team use
- Same API as cloud alternatives — easy to swap later

For a portfolio project and small-scale use, ChromaDB is the pragmatic choice. If this were a production multi-user system, switching to Pinecone would be a one-line change in `ingestor.py`.

---

## 4. Why nomic-embed-text for embeddings?

**nomic-embed-text** is a purpose-built embedding model optimized for text retrieval tasks. Compared to using the LLM itself for embeddings:
- Much faster (embedding models are smaller than LLMs)
- Better semantic similarity for retrieval tasks
- Runs locally via Ollama — consistent with our no-cloud approach

The alternative would be OpenAI's `text-embedding-ada-002`, but that requires an API key and costs money per embedding call.

---

## 5. Why chunk size 500 with 50-character overlap?

**Chunk size** controls how much text goes into each vector. Too large → each chunk contains too many topics, making retrieval imprecise. Too small → chunks lose context and answers become fragmented.

**500 characters** (~80-100 words) is a good middle ground for most document types — large enough to contain a complete idea, small enough to be specific.

**50-character overlap** ensures sentences that span chunk boundaries don't lose context. If a key sentence starts at the end of chunk 3 and continues into chunk 4, both chunks contain it.

---

## 6. Why FastAPI instead of Flask?

**Flask** is simpler but requires manual input validation and doesn't generate API documentation automatically.

**FastAPI** gives:
- Automatic request validation via Pydantic models
- Auto-generated interactive API docs at `/docs`
- Native async support for future scalability
- Type hints throughout — cleaner, more maintainable code

The auto-generated `/docs` page alone makes FastAPI worth it — it lets anyone test the API without writing a single line of client code.

---

## 7. Why Streamlit for the UI?

The alternative was building a React frontend. For a backend-focused portfolio project, spending a week on React would distract from the core engineering work.

**Streamlit** lets you build a functional, good-looking web UI in pure Python:
- No HTML, CSS, or JavaScript needed
- Built-in chat components, file uploaders, and session state
- Fast to iterate — changes reflect immediately on save
- Sufficient for demonstrating the product in interviews

If this were a production product, a React frontend calling the FastAPI backend would be the right move. The API is already built to support that migration.
