"""End-to-end tests for the retrieval evaluation path and results table."""

from __future__ import annotations

from pathlib import Path

from mizan.config import ModelSpec, RetrieverSpec, RunConfig
from mizan.report import retrieval_table
from mizan.results import RunMetadata
from mizan.runner import run_eval
from mizan.schemas import EvalItem, GoldLabel, ItemVariant, Language, TaskType

CORPUS = "\n".join(
    [
        '{"id": "d-riyadh", "title": "الرياض", "text": "الرياض هي عاصمة السعودية."}',
        '{"id": "d-cairo", "title": "القاهرة", "text": "القاهرة هي عاصمة مصر."}',
        '{"id": "d-doha", "title": "الدوحة", "text": "الدوحة هي عاصمة قطر."}',
    ]
)


def _retrieval_item() -> EvalItem:
    return EvalItem(
        id="r1",
        task_type=TaskType.RETRIEVAL,
        variants={
            Language.EN: ItemVariant(query="What is the capital of Saudi Arabia?"),
            Language.MSA: ItemVariant(query="ما هي عاصمة السعودية؟"),
            Language.GULF: ItemVariant(query="وش عاصمة السعودية؟"),
        },
        gold=GoldLabel(relevant_doc_ids=["d-riyadh"]),
    )


def _config(corpus: str, *, with_retriever: bool) -> RunConfig:
    return RunConfig(
        name="ret",
        dataset="d.jsonl",
        models=[ModelSpec(id="mock")],
        tasks=[TaskType.RETRIEVAL],
        languages=[Language.EN, Language.MSA, Language.GULF],
        retrieval_k=[1, 3],
        corpus=corpus if with_retriever else None,
        retrievers=[RetrieverSpec(id="bm25")] if with_retriever else [],
    )


def _metadata(config: RunConfig) -> RunMetadata:
    return RunMetadata(
        run_id="r",
        name="ret",
        created_at="2026-07-03T00:00:00+00:00",
        mizan_version="0.1.0",
        model_ids=config.model_ids,
        config=config,
    )


def test_retrieval_scores_are_computed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(CORPUS, encoding="utf-8")
    config = _config(str(corpus), with_retriever=True)

    result = run_eval(config, [_retrieval_item()], _metadata(config), cache_root=tmp_path / "c")

    # 1 item x 3 languages x 1 retriever.
    assert len(result.items) == 3
    msa = next(r for r in result.items if r.language is Language.MSA)
    assert msa.model_id == "bm25"
    assert msa.scores["recall@1"] == 1.0
    assert msa.scores["mrr"] == 1.0
    assert set(msa.scores) == {"recall@1", "recall@3", "mrr"}


def test_retrieval_table_renders(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(CORPUS, encoding="utf-8")
    config = _config(str(corpus), with_retriever=True)
    result = run_eval(config, [_retrieval_item()], _metadata(config), cache_root=tmp_path / "c")

    table = retrieval_table(result)
    assert "| System | Language |" in table
    assert "bm25" in table
    assert "MRR" in table
    assert "msa" in table


def test_no_retriever_marks_items_pending(tmp_path: Path) -> None:
    config = _config("", with_retriever=False)
    result = run_eval(config, [_retrieval_item()], _metadata(config), cache_root=tmp_path / "c")
    assert result.items == []
    assert result.summary["retrieval_pending"] == 1


def test_table_placeholder_when_no_retrieval(tmp_path: Path) -> None:
    config = _config("", with_retriever=False)
    result = run_eval(config, [_retrieval_item()], _metadata(config), cache_root=tmp_path / "c")
    assert "No retrieval results" in retrieval_table(result)
