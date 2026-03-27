import requests
import httpx
import sys
import pytest

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test pentru endpoint-ul root GET /"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "RAG Assistant" in data["message"]


def test_chat_positive_relevant_question():
    """Test pozitiv: intrebare relevanta despre VAIP evaluata cu LLM-as-a-Judge."""
    response = requests.post(
        f"{BASE_URL}/chat/",
        json={"message": "Cum detecteaza platforma VAIP coroziunea pe echipamentele industriale?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    answer = data["response"]
    assert len(answer) > 50, "Raspunsul trebuie sa fie detaliat"
    assert not answer.startswith("Intrebarea ta nu pare"), "Raspunsul nu ar trebui sa fie respins ca irelevant"


def test_chat_negative_irrelevant_question():
    """Test negativ: intrebare irelevanta respinsa de asistent."""
    response = requests.post(
        f"{BASE_URL}/chat/",
        json={"message": "Care este cea mai buna reteta de pizza cu ananas?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    answer = data["response"]
    assert "nu pare" in answer.lower() or "irelevant" in answer.lower() or "VAIP" in answer, \
        "Asistentul ar trebui sa respinga intrebarea irelevanta"
