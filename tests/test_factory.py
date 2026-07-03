"""Tests for the backend factory."""

from __future__ import annotations

import pytest

from mizan.config import ModelSpec
from mizan.llm.base import BackendError
from mizan.llm.factory import build_backend
from mizan.llm.huggingface import HuggingFaceBackend
from mizan.llm.mock import MockBackend
from mizan.llm.openai_compatible import OpenAICompatibleBackend


def test_builds_mock_backend() -> None:
    backend = build_backend(ModelSpec(id="m1", backend="mock"), env={})
    assert isinstance(backend, MockBackend)
    assert backend.model_id == "m1"


def test_builds_openai_backend_with_default_key() -> None:
    spec = ModelSpec(id="gpt-5.5-2026-04-23", backend="openai")
    backend = build_backend(spec, env={"OPENAI_API_KEY": "sk-x"})
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model_id == "gpt-5.5-2026-04-23"


def test_builds_openai_backend_with_custom_provider() -> None:
    spec = ModelSpec(
        id="Qwen/Qwen2.5-7B-Instruct",
        backend="openai",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
    )
    backend = build_backend(spec, env={"DEEPINFRA_API_KEY": "di-x"})
    assert isinstance(backend, OpenAICompatibleBackend)


def test_openai_missing_key_names_the_variable() -> None:
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        build_backend(ModelSpec(id="gpt", backend="openai"), env={})


def test_openai_missing_custom_key_names_that_variable() -> None:
    spec = ModelSpec(id="q", backend="openai", api_key_env="DEEPINFRA_API_KEY")
    with pytest.raises(BackendError, match="DEEPINFRA_API_KEY"):
        build_backend(spec, env={})


def test_builds_hf_backend_with_revision() -> None:
    spec = ModelSpec(id="humain-ai/ALLaM-7B-Instruct-preview", backend="hf", revision="abc123")
    backend = build_backend(spec, env={"HF_TOKEN": "hf-x"})
    assert isinstance(backend, HuggingFaceBackend)
    assert backend.revision == "abc123"


def test_unknown_backend_raises() -> None:
    with pytest.raises(BackendError, match="unknown backend"):
        build_backend(ModelSpec(id="m", backend="local"), env={})
