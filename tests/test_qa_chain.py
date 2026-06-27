import sys
import os
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_chain import build_prompt


# --- build_prompt tests ---

def test_build_prompt_contains_question():
    chunks = ["Python is a programming language."]
    prompt = build_prompt("What is Python?", chunks)
    assert "What is Python?" in prompt


def test_build_prompt_contains_chunk_text():
    chunks = ["FastAPI is a modern web framework."]
    prompt = build_prompt("What is FastAPI?", chunks)
    assert "FastAPI is a modern web framework." in prompt


def test_build_prompt_contains_multiple_chunks():
    chunks = ["Chunk one content.", "Chunk two content.", "Chunk three content."]
    prompt = build_prompt("Test question?", chunks)
    assert "Chunk one content." in prompt
    assert "Chunk two content." in prompt
    assert "Chunk three content." in prompt


def test_build_prompt_returns_string():
    chunks = ["Some context."]
    prompt = build_prompt("Question?", chunks)
    assert isinstance(prompt, str)


def test_build_prompt_not_empty():
    chunks = ["Context here."]
    prompt = build_prompt("Question?", chunks)
    assert len(prompt) > 0


def test_build_prompt_instructs_to_use_context():
    chunks = ["Some text."]
    prompt = build_prompt("Question?", chunks)
    assert "context" in prompt.lower()


def test_build_prompt_empty_chunks():
    prompt = build_prompt("Question?", [])
    assert isinstance(prompt, str)


# --- answer_question tests (mocked) ---

def test_answer_question_returns_dict():
    mock_response = {"message": {"content": "Python is a programming language."}}
    with patch("qa_chain.get_relevant_chunks", return_value=["Python is great."]):
        with patch("qa_chain.ollama.chat", return_value=mock_response):
            from qa_chain import answer_question
            result = answer_question("What is Python?", "fake_collection")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result

def test_answer_question_returns_correct_answer():
    mock_response = {"message": {"content": "Django is a web framework for Python."}}
    with patch("qa_chain.get_relevant_chunks", return_value=["Django is a Python web framework."]):
        with patch("qa_chain.ollama.chat", return_value=mock_response):
            from qa_chain import answer_question
            result = answer_question("What is Django?", "fake_collection")
    assert result["answer"] == "Django is a web framework for Python."

def test_answer_question_no_chunks_returns_fallback():
    with patch("qa_chain.get_relevant_chunks", return_value=[]):
        from qa_chain import answer_question
        result = answer_question("Random question?", "fake_collection")
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result["sources"] == []

def test_answer_question_sources_match_retrieved_chunks():
    mock_response = {"message": {"content": "Some answer."}}
    with patch("qa_chain.get_relevant_chunks", return_value=["Chunk A", "Chunk B"]):
        with patch("qa_chain.ollama.chat", return_value=mock_response):
            from qa_chain import answer_question
            result = answer_question("Question?", "fake_collection")
    assert len(result["sources"]) == 2