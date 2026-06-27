import streamlit as st
import requests
import tempfile
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Powered by Ollama + ChromaDB")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Reading and embedding your document..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["collection_name"] = data["collection_name"]
                        st.session_state["filename"] = uploaded_file.name
                        st.session_state["messages"] = []
                        st.success(f"Ingested {data['chunk_count']} chunks!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

    if "filename" in st.session_state:
        st.divider()
        st.caption(f"Active document:")
        st.info(st.session_state["filename"])

        if st.button("Clear document"):
            for key in ["collection_name", "filename", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

# --- Main: Chat Interface ---
st.title("Ask anything about your document")

if "collection_name" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View source chunks"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)

# Chat input
question = st.chat_input("Ask a question about your document...")

if question:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "collection_name": st.session_state["collection_name"],
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    st.markdown(answer)
                    with st.expander("View source chunks"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    error = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")