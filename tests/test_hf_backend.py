"""Tests for the Hugging Face backend using a fake injected client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mizan.llm.base import BackendError, GenerationParams
from mizan.llm.huggingface import HuggingFaceBackend


class FakeHFClient:
    def __init__(self, content: str = "مرحبا", raises: Exception | None = None) -> None:
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_returns_content_and_records_revision() -> None:
    fake = FakeHFClient(content="الجواب")
    backend = HuggingFaceBackend(
        "humain-ai/ALLaM-7B-Instruct-preview", client=fake, revision="abc123"
    )
    assert backend.generate("سؤال", GenerationParams()) == "الجواب"
    assert backend.revision == "abc123"
    assert fake.calls[0]["model"] == "humain-ai/ALLaM-7B-Instruct-preview"


def test_temperature_zero_is_omitted() -> None:
    fake = FakeHFClient()
    HuggingFaceBackend("m", client=fake).generate("q", GenerationParams(temperature=0.0))
    assert "temperature" not in fake.calls[0]


def test_temperature_and_seed_passed_when_set() -> None:
    fake = FakeHFClient()
    HuggingFaceBackend("m", client=fake).generate("q", GenerationParams(temperature=0.5, seed=3))
    assert fake.calls[0]["temperature"] == 0.5
    assert fake.calls[0]["seed"] == 3


def test_client_error_is_wrapped() -> None:
    fake = FakeHFClient(raises=RuntimeError("503 service unavailable"))
    with pytest.raises(BackendError, match="inference call failed"):
        HuggingFaceBackend("m", client=fake).generate("q", GenerationParams())
