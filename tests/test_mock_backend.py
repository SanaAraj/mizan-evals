"""Tests for the deterministic mock backend."""

from __future__ import annotations

from mizan.llm.base import GenerationParams
from mizan.llm.mock import MockBackend


def test_output_is_deterministic() -> None:
    backend = MockBackend()
    params = GenerationParams()
    first = backend.generate("مرحبا", params)
    second = backend.generate("مرحبا", params)
    assert first == second


def test_different_prompts_differ() -> None:
    backend = MockBackend()
    params = GenerationParams()
    assert backend.generate("a", params) != backend.generate("b", params)


def test_seed_changes_output() -> None:
    backend = MockBackend()
    assert backend.generate("q", GenerationParams(seed=1)) != backend.generate(
        "q", GenerationParams(seed=2)
    )


def test_scripted_responses_take_priority() -> None:
    backend = MockBackend(scripted={"ping": "pong"})
    assert backend.generate("ping", GenerationParams()) == "pong"


def test_calls_counter_increments() -> None:
    backend = MockBackend()
    params = GenerationParams()
    backend.generate("a", params)
    backend.generate("b", params)
    assert backend.calls == 2


def test_model_id_is_reported() -> None:
    assert MockBackend(model_id="allam-7b").model_id == "allam-7b"
