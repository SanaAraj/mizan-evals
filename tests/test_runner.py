"""Tests for the evaluation runner loop."""

from __future__ import annotations

from pathlib import Path

from mizan.config import ModelSpec, RunConfig
from mizan.llm.base import Backend, BackendError, GenerationParams
from mizan.results import RunMetadata
from mizan.runner import run_eval
from mizan.schemas import (
    EvalItem,
    ExpectedToolCall,
    GoldLabel,
    ItemVariant,
    Language,
    TaskType,
)


def _items() -> list[EvalItem]:
    trilingual = {
        Language.EN: ItemVariant(query="q-en", reference_answer="a"),
        Language.MSA: ItemVariant(query="q-msa", reference_answer="a"),
        Language.GULF: ItemVariant(query="q-gulf", reference_answer="a"),
    }
    return [
        EvalItem(id="aq", task_type=TaskType.ANSWER_QUALITY, variants=trilingual),
        EvalItem(
            id="tc",
            task_type=TaskType.TOOL_CALLING,
            variants={k: ItemVariant(query=v.query) for k, v in trilingual.items()},
            gold=GoldLabel(expected_tool=ExpectedToolCall(name="get_weather")),
        ),
        EvalItem(
            id="ret",
            task_type=TaskType.RETRIEVAL,
            variants={k: ItemVariant(query=v.query) for k, v in trilingual.items()},
            gold=GoldLabel(relevant_doc_ids=["d1"]),
        ),
    ]


def _config() -> RunConfig:
    return RunConfig(
        name="t",
        dataset="d.jsonl",
        models=[ModelSpec(id="mock-a"), ModelSpec(id="mock-b")],
        tasks=[TaskType.ANSWER_QUALITY, TaskType.TOOL_CALLING, TaskType.RETRIEVAL],
        languages=[Language.EN, Language.MSA],
    )


def _metadata(config: RunConfig) -> RunMetadata:
    return RunMetadata(
        run_id="r",
        name="t",
        created_at="2026-07-03T00:00:00+00:00",
        mizan_version="0.1.0",
        model_ids=config.model_ids,
        config=config,
    )


def test_runs_generation_tasks_across_selected_languages(tmp_path: Path) -> None:
    config = _config()
    result = run_eval(config, _items(), _metadata(config), cache_root=tmp_path)

    # 2 generation items x 2 selected languages x 2 models = 8 results.
    assert len(result.items) == 8
    assert all(r.output is not None for r in result.items)
    # Gulf was not selected, so no result should be in Gulf.
    assert all(r.language != Language.GULF for r in result.items)
    # The retrieval item has no configured retriever; it is counted as pending.
    assert result.summary["retrieval_pending"] == 1
    assert result.summary["n_results"] == 8
    assert result.summary["n_errors"] == 0


def test_second_run_is_fully_cached(tmp_path: Path) -> None:
    config = _config()
    run_eval(config, _items(), _metadata(config), cache_root=tmp_path)
    second = run_eval(config, _items(), _metadata(config), cache_root=tmp_path)
    assert second.summary["n_cached"] == 8
    assert all(r.cached for r in second.items)


def test_backend_errors_are_recorded_per_item(tmp_path: Path) -> None:
    class FailingBackend(Backend):
        @property
        def model_id(self) -> str:
            return "fail"

        def generate(self, prompt: str, params: GenerationParams) -> str:
            raise BackendError("boom")

    config = _config()
    result = run_eval(
        config,
        _items(),
        _metadata(config),
        cache_root=tmp_path,
        backend_builder=lambda spec, env: FailingBackend(),
    )
    assert result.summary["n_errors"] == 8
    assert all(r.output is None and r.error == "boom" for r in result.items)
