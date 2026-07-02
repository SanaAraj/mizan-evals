"""Tests for the disk-cached, resumable LLM client."""

from __future__ import annotations

from pathlib import Path

from mizan.llm.base import GenerationParams
from mizan.llm.client import CachedLLMClient
from mizan.llm.mock import MockBackend


def test_first_call_misses_then_second_hits(tmp_path: Path) -> None:
    backend = MockBackend()
    client = CachedLLMClient(backend, tmp_path)

    first = client.generate("what is 2+2?")
    assert first.cached is False
    assert backend.calls == 1

    second = client.generate("what is 2+2?")
    assert second.cached is True
    assert second.text == first.text
    # The cache hit must not have invoked the backend again.
    assert backend.calls == 1


def test_distinct_prompts_are_cached_separately(tmp_path: Path) -> None:
    backend = MockBackend()
    client = CachedLLMClient(backend, tmp_path)
    client.generate("a")
    client.generate("b")
    assert backend.calls == 2


def test_distinct_params_are_cached_separately(tmp_path: Path) -> None:
    backend = MockBackend()
    client = CachedLLMClient(backend, tmp_path)
    client.generate("q", GenerationParams(temperature=0.0))
    client.generate("q", GenerationParams(temperature=0.7))
    assert backend.calls == 2


def test_cache_persists_across_client_instances(tmp_path: Path) -> None:
    # Simulates resuming an interrupted run: a fresh client + fresh backend must
    # serve the earlier result from disk without re-invoking the backend.
    CachedLLMClient(MockBackend(), tmp_path).generate("resume me")

    new_backend = MockBackend()
    resumed = CachedLLMClient(new_backend, tmp_path).generate("resume me")
    assert resumed.cached is True
    assert new_backend.calls == 0


def test_arabic_prompt_is_cached_and_readable(tmp_path: Path) -> None:
    client = CachedLLMClient(MockBackend(), tmp_path)
    client.generate("ما هي عاصمة قطر؟")
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    # Cache is written with ensure_ascii=False so it stays human-reviewable.
    assert "ما هي عاصمة قطر؟" in cache_files[0].read_text(encoding="utf-8")


def test_corrupt_cache_entry_is_recomputed(tmp_path: Path) -> None:
    backend = MockBackend()
    client = CachedLLMClient(backend, tmp_path)
    client.generate("q")
    assert backend.calls == 1

    # Corrupt the single cache file, as an interrupted write might.
    cache_file = next(iter(tmp_path.glob("*.json")))
    cache_file.write_text("{ not valid json", encoding="utf-8")

    result = client.generate("q")
    assert result.cached is False
    assert backend.calls == 2
