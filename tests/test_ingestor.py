import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestor import chunk_text, get_collection_name


# --- chunk_text tests ---

def test_chunk_text_returns_list():
    """Should return a list of strings."""
    text = "This is a sample document. " * 50
    chunks = chunk_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_splits_long_text():
    """Long text should be split into multiple chunks."""
    text = "word " * 500
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunk_text_short_text_single_chunk():
    """Short text that fits in one chunk should return one chunk."""
    text = "This is a short document."
    chunks = chunk_text(text)
    assert len(chunks) == 1


def test_chunk_text_chunks_are_strings():
    """Each chunk should be a string."""
    text = "This is a test document. " * 100
    chunks = chunk_text(text)
    for chunk in chunks:
        assert isinstance(chunk, str)


def test_chunk_text_overlap_means_content_shared():
    """With overlap, adjacent chunks should share some content."""
    text = "The quick brown fox jumps over the lazy dog. " * 50
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        words_first = set(chunks[0].split())
        words_second = set(chunks[1].split())
        assert len(words_first & words_second) > 0


def test_chunk_text_empty_string():
    """Empty string should return empty list."""
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_text_preserves_content():
    """All words from original text should appear somewhere in chunks."""
    text = "Python FastAPI ChromaDB Ollama LangChain " * 10
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    for word in ["Python", "FastAPI", "ChromaDB", "Ollama", "LangChain"]:
        assert word in combined


# --- get_collection_name tests ---

def test_get_collection_name_returns_string():
    """Should return a string."""
    name = get_collection_name("test.pdf")
    assert isinstance(name, str)


def test_get_collection_name_same_file_same_name():
    """Same filename should always produce the same collection name."""
    name1 = get_collection_name("document.pdf")
    name2 = get_collection_name("document.pdf")
    assert name1 == name2


def test_get_collection_name_different_files_different_names():
    """Different filenames should produce different collection names."""
    name1 = get_collection_name("doc1.pdf")
    name2 = get_collection_name("doc2.pdf")
    assert name1 != name2


def test_get_collection_name_no_special_chars():
    """Collection name should only contain alphanumeric chars and underscores."""
    name = get_collection_name("my document (final).pdf")
    for char in name:
        assert char.isalnum() or char == "_", f"Invalid char: {char}"


def test_get_collection_name_handles_spaces():
    """Should handle filenames with spaces."""
    name = get_collection_name("Business Report 2024.pdf")
    assert isinstance(name, str)
    assert len(name) > 0