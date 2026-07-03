"""The evaluation loop: iterate systems x items x languages and record results.

Two kinds of system are evaluated. LLM backends (answer-quality, faithfulness,
tool-calling) generate text through :class:`CachedLLMClient`, so interrupted runs
resume from disk. Retrievers score ``recall@k``/``MRR`` over the corpus. When no
retriever is configured, retrieval items are recorded as pending rather than
silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from mizan.config import ModelSpec, RunConfig
from mizan.corpus import Document, load_documents
from mizan.llm.base import Backend, BackendError, GenerationParams
from mizan.llm.client import CachedLLMClient
from mizan.llm.factory import build_backend
from mizan.results import ItemResult, RunMetadata, RunResult
from mizan.retrieval.factory import build_retriever
from mizan.schemas import EvalItem, Language, TaskType
from mizan.scoring.retrieval import recall_at_k, reciprocal_rank

BackendBuilder = Callable[[ModelSpec, Mapping[str, str] | None], Backend]

# Tasks handled by generating raw model output. Retrieval is scored separately.
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


def _evaluate_generation_item(
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
        response = client.generate(item.variants[language].query, params)
    except BackendError as exc:
        return ItemResult(**base, output=None, error=str(exc))
    return ItemResult(**base, output=response.text, cached=response.cached)


def _run_generation(
    config: RunConfig,
    items: list[EvalItem],
    cache_root: Path,
    env: Mapping[str, str] | None,
    backend_builder: BackendBuilder,
) -> list[ItemResult]:
    selected_tasks = set(config.tasks) & _GENERATION_TASKS
    if not selected_tasks:
        return []
    selected_langs = set(config.languages)
    results: list[ItemResult] = []
    for spec in config.models:
        backend = backend_builder(spec, env)
        client = CachedLLMClient(backend, cache_root / _safe_dirname(spec.id))
        params = _generation_params(spec, config.seed)
        for item in items:
            if item.task_type not in selected_tasks:
                continue
            for language in item.languages:
                if language in selected_langs:
                    results.append(_evaluate_generation_item(client, spec, params, item, language))
    return results


def _run_retrieval(
    config: RunConfig,
    items: list[EvalItem],
    documents: list[Document],
) -> list[ItemResult]:
    ks = sorted(set(config.retrieval_k))
    top_k = ks[-1]
    selected_langs = set(config.languages)
    retrieval_items = [i for i in items if i.task_type is TaskType.RETRIEVAL]
    results: list[ItemResult] = []
    for spec in config.retrievers:
        retriever = build_retriever(spec, documents)
        for item in retrieval_items:
            gold = item.gold.relevant_doc_ids
            for language in item.languages:
                if language not in selected_langs:
                    continue
                retrieved = retriever.retrieve(item.variants[language].query, top_k)
                scores = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in ks}
                scores["mrr"] = reciprocal_rank(retrieved, gold)
                results.append(
                    ItemResult(
                        item_id=item.id,
                        model_id=spec.id,
                        language=language,
                        task_type=TaskType.RETRIEVAL,
                        output=None,
                        scores=scores,
                    )
                )
    return results


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
    results = _run_generation(config, items, cache_root, env, backend_builder)

    pending_retrieval = 0
    if TaskType.RETRIEVAL in set(config.tasks):
        if config.retrievers and config.corpus:
            documents = load_documents(config.corpus)
            results += _run_retrieval(config, items, documents)
        else:
            pending_retrieval = sum(1 for i in items if i.task_type is TaskType.RETRIEVAL)

    return RunResult(
        metadata=metadata, items=results, summary=summarize(results, pending_retrieval)
    )


def summarize(results: list[ItemResult], pending_retrieval: int = 0) -> dict:
    """Produce a small aggregate summary of a run's item results."""
    by_system: dict[str, int] = {}
    for result in results:
        by_system[result.model_id] = by_system.get(result.model_id, 0) + 1
    return {
        "n_results": len(results),
        "n_errors": sum(1 for r in results if r.error is not None),
        "n_cached": sum(1 for r in results if r.cached),
        "by_system": by_system,
        "retrieval_pending": pending_retrieval,
    }
