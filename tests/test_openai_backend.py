"""Tests for the OpenAI-compatible backend using a mocked httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest

from mizan.llm.base import BackendError, GenerationParams
from mizan.llm.openai_compatible import OpenAICompatibleBackend


def _backend(handler, model_id: str = "gpt-x") -> OpenAICompatibleBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleBackend(
        model_id, base_url="https://api.test/v1", api_key="sk-test", client=client
    )


def test_sends_expected_request_and_parses_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "الرياض"}}]})

    backend = _backend(handler)
    text = backend.generate("ما هي عاصمة السعودية؟", GenerationParams(temperature=0.2, seed=7))

    assert text == "الرياض"
    assert seen["url"] == "https://api.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-x"
    assert seen["body"]["messages"] == [{"role": "user", "content": "ما هي عاصمة السعودية؟"}]
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["seed"] == 7


def test_seed_omitted_when_none() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _backend(handler).generate("hi", GenerationParams(seed=None))
    assert "seed" not in seen["body"]


def test_non_200_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with pytest.raises(BackendError, match="HTTP 429"):
        _backend(handler).generate("hi", GenerationParams())


def test_malformed_response_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(BackendError, match="unexpected response shape"):
        _backend(handler).generate("hi", GenerationParams())


def test_null_content_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    with pytest.raises(BackendError, match="no content"):
        _backend(handler).generate("hi", GenerationParams())
