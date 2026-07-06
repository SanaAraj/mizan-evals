"""End-to-end tests for the tool-calling path of the runner."""

from __future__ import annotations

from pathlib import Path

from mizan.config import ModelSpec, RunConfig
from mizan.llm.base import Backend, BackendError, GenerationParams
from mizan.report import tool_calling_table
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
from mizan.tools.extract import to_canonical_json

WEATHER = GoldLabel(
    expected_tool=ExpectedToolCall(name="get_weather", arguments={"location": "Riyadh"})
)


def _items() -> list[EvalItem]:
    return [
        EvalItem(
            id="tc-pos",
            task_type=TaskType.TOOL_CALLING,
            variants={
                Language.EN: ItemVariant(query="weather in Riyadh?"),
                Language.MSA: ItemVariant(query="الطقس في الرياض؟"),
                Language.GULF: ItemVariant(query="شلون الجو بالرياض؟"),
            },
            gold=WEATHER,
        ),
        EvalItem(
            id="tc-dist",
            task_type=TaskType.TOOL_CALLING,
            variants={
                Language.EN: ItemVariant(query="what do you think of my haircut?"),
                Language.MSA: ItemVariant(query="ما رأيك في قصة شعري؟"),
                Language.GULF: ItemVariant(query="شرايك بقصة شعري؟"),
            },
            gold=GoldLabel(expected_no_tool=True),
        ),
    ]


class ScriptedNativeBackend(Backend):
    """A native backend returning a canned decision per utterance."""

    def __init__(self, mapping: dict[str, tuple[str | None, dict]]) -> None:
        self._mapping = mapping

    @property
    def model_id(self) -> str:
        return "scripted-native"

    @property
    def supports_native_tools(self) -> bool:
        return True

    def generate(self, prompt: str, params: GenerationParams) -> str:  # pragma: no cover
        raise BackendError("native backend should not be prompted")

    def generate_tool_call(
        self, utterance: str, tools: list[dict], params: GenerationParams
    ) -> str:
        tool, args = self._mapping.get(utterance, (None, {}))
        return to_canonical_json(tool, args)


def _config(tool_mode: str = "auto") -> RunConfig:
    return RunConfig(
        name="tc",
        dataset="d.jsonl",
        models=[ModelSpec(id="m", tool_mode=tool_mode)],
        tasks=[TaskType.TOOL_CALLING],
        languages=[Language.EN, Language.MSA, Language.GULF],
    )


def _metadata(config: RunConfig) -> RunMetadata:
    return RunMetadata(
        run_id="r",
        name="tc",
        created_at="2026-07-06T00:00:00+00:00",
        mizan_version="0.1.0",
        model_ids=config.model_ids,
        config=config,
    )


def _run(mapping: dict[str, tuple[str | None, dict]], tmp_path: Path):
    config = _config()
    return run_eval(
        config,
        _items(),
        _metadata(config),
        cache_root=tmp_path,
        backend_builder=lambda spec, env: ScriptedNativeBackend(mapping),
    )


def test_scores_correct_localized_and_wrong_calls(tmp_path: Path) -> None:
    mapping = {
        "weather in Riyadh?": ("get_weather", {"location": "Riyadh"}),  # correct, english
        "الطقس في الرياض؟": ("get_weather", {"location": "الرياض"}),  # correct, localized
        "شلون الجو بالرياض؟": ("web_search", {"query": "weather"}),  # wrong tool
        # distractor utterances default to no-call (correct refusal)
    }
    result = _run(mapping, tmp_path)
    by = {(r.item_id, r.language): r for r in result.items}

    en = by[("tc-pos", Language.EN)]
    assert en.scores["tool_correct"] == 1.0
    assert en.scores["arg_accuracy"] == 1.0
    assert en.scores["localized_correct"] == 0.0

    msa = by[("tc-pos", Language.MSA)]
    assert msa.scores["tool_correct"] == 1.0
    assert msa.scores["localized_correct"] == 1.0  # الرياض mapped to Riyadh

    gulf = by[("tc-pos", Language.GULF)]
    assert gulf.scores["tool_correct"] == 0.0
    assert "arg_accuracy" not in gulf.scores

    # Distractors: no call -> correct refusal in every language.
    for lang in (Language.EN, Language.MSA, Language.GULF):
        assert by[("tc-dist", lang)].scores["refused"] == 1.0


def test_hallucinated_and_failed_calls(tmp_path: Path) -> None:
    mapping = {
        "weather in Riyadh?": ("teleport", {}),  # invented tool
        "ما رأيك في قصة شعري؟": ("get_weather", {"location": "x"}),  # distractor, false tool call
    }
    result = _run(mapping, tmp_path)
    by = {(r.item_id, r.language): r for r in result.items}

    assert by[("tc-pos", Language.EN)].scores["hallucinated_tool"] == 1.0
    assert by[("tc-pos", Language.EN)].chosen_tool == "teleport"
    # A distractor that called a real tool did not refuse.
    assert by[("tc-dist", Language.MSA)].scores["refused"] == 0.0


def test_second_run_is_cached(tmp_path: Path) -> None:
    mapping = {"weather in Riyadh?": ("get_weather", {"location": "Riyadh"})}
    _run(mapping, tmp_path)
    second = _run(mapping, tmp_path)
    assert all(r.cached for r in second.items)


def test_table_renders_all_languages(tmp_path: Path) -> None:
    mapping = {"weather in Riyadh?": ("get_weather", {"location": "Riyadh"})}
    result = _run(mapping, tmp_path)
    table = tool_calling_table(result)
    assert "correct-tool" in table
    assert "localization" in table
    # one header + separator + (2 items aggregated per language) = 3 language rows
    assert table.count("| m | ") == 3


class FailingBackend(Backend):
    @property
    def model_id(self) -> str:
        return "fail"

    def generate(self, prompt: str, params: GenerationParams) -> str:
        raise BackendError("boom")

    @property
    def supports_native_tools(self) -> bool:
        return False


def test_backend_error_is_recorded_and_not_a_refusal(tmp_path: Path) -> None:
    config = _config()
    result = run_eval(
        config,
        _items(),
        _metadata(config),
        cache_root=tmp_path,
        backend_builder=lambda spec, env: FailingBackend(),
    )
    dist = next(r for r in result.items if r.item_id == "tc-dist" and r.language is Language.EN)
    assert dist.error == "boom"
    assert dist.output is None
    # A transport failure must not be counted as a correct refusal.
    assert dist.scores["refused"] == 0.0
