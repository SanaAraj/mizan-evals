"""The evaluation loop: iterate models x items x languages and record results.

Every generation goes through :class:`CachedLLMClient`, so an interrupted run
resumes from disk without recomputation. Phase 1 wires the LLM-generation path
(answer-quality, faithfulness, tool-calling raw outputs); retrieval scoring and
task-specific metrics arrive in later phases, so retrieval items are recorded as
pending rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from mizan.config import ModelSpec, RunConfig
from mizan.llm.base import Backend, BackendError, GenerationParams
from mizan.llm.client import CachedLLMClient
from mizan.llm.factory import build_backend
from mizan.results import ItemResult, RunMetadata, RunResult
from mizan.schemas import EvalItem, Language, TaskType

BackendBuilder = Callable[[ModelSpec, Mapping[str, str] | None], Backend]

# Tasks whose Phase-1 handling is "generate raw model output". Retrieval is a
# retriever task (Phase 2) and is not driven through an LLM backend.
_GENERATION_TASKS = frozenset(
    {TaskType.ANSWER_QUALITY, TaskType.FAITHFULNESS, TaskType.TOOL_CALLING}
)


def _safe_dirname(model_id: str) -> str:
    """Turn a model id such as ``Qwen/Qwen2.5-7B-Instruct`` into a safe dir name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)


def _generation_params(spec: ModelSpec, default_seed: int) -> GenerationParams:
    params = spec.params
    return GenerationParams(
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_tokens", 512),
        seed=params.get("seed", default_seed),
    )


def _build_prompt(item: EvalItem, language: Language) -> str:
    """Build the model prompt for one item variant.

    Phase 1 uses the bare query. Retrieved context (RAG) and tool schemas are
    layered on in later phases.
    """
    return item.variants[language].query


def _evaluate_item(
    client: CachedLLMClient,
    spec: ModelSpec,
    params: GenerationParams,
    item: EvalItem,
    language: Language,
) -> ItemResult:
    base = {
        "item_id": item.id,
        "model_id": spec.id,
        "language": language,
        "task_type": item.task_type,
    }
    try:
        response = client.generate(_build_prompt(item, language), params)
    except BackendError as exc:
        return ItemResult(**base, output=None, error=str(exc))
    return ItemResult(**base, output=response.text, cached=response.cached)


def run_eval(
    config: RunConfig,
    items: list[EvalItem],
    metadata: RunMetadata,
    *,
    cache_root: Path,
    env: Mapping[str, str] | None = None,
    backend_builder: BackendBuilder = build_backend,
) -> RunResult:
    """Execute the evaluation and return a :class:`RunResult`.

    Args:
        config: the validated run configuration.
        items: the loaded evaluation items.
        metadata: run metadata to embed in the result.
        cache_root: root directory for the disk cache (one subdir per model).
        env: environment mapping for backend keys (defaults to ``os.environ``).
        backend_builder: injectable backend factory (for tests).
    """
    selected_tasks = set(config.tasks)
    selected_langs = set(config.languages)
    results: list[ItemResult] = []
    pending_retrieval = 0

    for spec in config.models:
        backend = backend_builder(spec, env)
        client = CachedLLMClient(backend, cache_root / _safe_dirname(spec.id))
        params = _generation_params(spec, config.seed)

        for item in items:
            if item.task_type not in selected_tasks:
                continue
            if item.task_type not in _GENERATION_TASKS:
                pending_retrieval += 1  # counted once per model for transparency
                continue
            for language in item.languages:
                if language not in selected_langs:
                    continue
                results.append(_evaluate_item(client, spec, params, item, language))

    return RunResult(
        metadata=metadata, items=results, summary=summarize(results, pending_retrieval)
    )


def summarize(results: list[ItemResult], pending_retrieval: int = 0) -> dict:
    """Produce a small aggregate summary of a run's item results."""
    by_model: dict[str, int] = {}
    for result in results:
        by_model[result.model_id] = by_model.get(result.model_id, 0) + 1
    return {
        "n_results": len(results),
        "n_errors": sum(1 for r in results if r.error is not None),
        "n_cached": sum(1 for r in results if r.cached),
        "by_model": by_model,
        "retrieval_pending": pending_retrieval,
    }
