"""
Unit tests for OllamaLLM client and LLMProvider protocol.
"""
import json
import pytest
import httpx

from apps.agent.llm import (
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMTimeoutError,
    OllamaLLM,
)
from apps.agent.models import ChatMessage


def test_ollama_llm_implements_protocol():
    llm = OllamaLLM(model_name="qwen3:8b")
    assert isinstance(llm, LLMProvider)
    assert llm.model_name == "qwen3:8b"


def test_ollama_chat_success():
    fake_response = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "The net payable claim amount is €22,950.00.",
            "thinking": "The user is asking about the net payable claim amount from C-1000.",
        },
        "prompt_eval_count": 45,
        "eval_count": 22,
        "total_duration": 1_500_000_000,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        req_body = json.loads(request.content.decode())
        assert req_body["model"] == "qwen3:8b"
        assert req_body["stream"] is False
        assert len(req_body["messages"]) == 2
        return httpx.Response(200, json=fake_response)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver")

    llm = OllamaLLM(model_name="qwen3:8b", client=client)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the net amount?"),
    ]

    resp = llm.chat(messages, temperature=0.0)
    assert resp.content == "The net payable claim amount is €22,950.00."
    assert resp.thinking == "The user is asking about the net payable claim amount from C-1000."
    assert resp.model == "qwen3:8b"
    assert resp.prompt_tokens == 45
    assert resp.completion_tokens == 22
    assert resp.duration_seconds == 1.5


def test_ollama_chat_inline_think_tag_extraction():
    fake_response = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "<think>\nEvaluate line items and deductible.\n</think>\nFinal answer: €22,950.00",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fake_response)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver")

    llm = OllamaLLM(model_name="qwen3:8b", client=client)
    resp = llm.chat([ChatMessage(role="user", content="Hello")])

    assert resp.content == "Final answer: €22,950.00"
    assert resp.thinking == "Evaluate line items and deductible."


def test_ollama_generate_convenience_method():
    def handler(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode())
        assert len(req_body["messages"]) == 2
        assert req_body["messages"][0]["role"] == "system"
        assert req_body["messages"][1]["role"] == "user"
        return httpx.Response(200, json={"model": "qwen3:8b", "message": {"content": "Generated answer"}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    llm = OllamaLLM(client=client)

    resp = llm.generate("Test prompt", system_prompt="System instructions")
    assert resp.content == "Generated answer"


def test_ollama_model_not_found_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model 'qwen3:8b' not found")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    llm = OllamaLLM(client=client)

    with pytest.raises(LLMModelNotFoundError) as exc_info:
        llm.chat([ChatMessage(role="user", content="Hi")])
    assert "not found" in str(exc_info.value).lower()


def test_ollama_server_error_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    llm = OllamaLLM(client=client)

    with pytest.raises(LLMError) as exc_info:
        llm.chat([ChatMessage(role="user", content="Hi")])
    assert "500" in str(exc_info.value)


def test_ollama_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    llm = OllamaLLM(client=client)

    with pytest.raises(LLMConnectionError):
        llm.chat([ChatMessage(role="user", content="Hi")])


def test_ollama_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Request timed out")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    llm = OllamaLLM(client=client)

    with pytest.raises(LLMTimeoutError):
        llm.chat([ChatMessage(role="user", content="Hi")])


def test_ollama_empty_messages_validation():
    llm = OllamaLLM()
    with pytest.raises(ValueError) as exc:
        llm.chat([])
    assert "empty" in str(exc.value)
