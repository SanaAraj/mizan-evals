"""Quality gates for the committed retrieval evaluation slice.

These tests pin the methodology claims made in ``data/retrieval/README.md``:
every item is parallel across the three languages, carries its review
provenance, and has gold ids that resolve against the pinned corpus.
"""

from __future__ import annotations

from pathlib import Path

from mizan.corpus import load_documents
from mizan.dataset import load_items
from mizan.schemas import Language, ReviewStatus, TaskType

REPO_ROOT = Path(__file__).resolve().parent.parent
SLICE = REPO_ROOT / "data" / "retrieval" / "eval_items.jsonl"
CORPUS = REPO_ROOT / "data" / "corpus" / "arwiki.jsonl"


def test_slice_items_are_parallel_retrieval_items() -> None:
    items = load_items(SLICE)
    assert len(items) == 15
    for item in items:
        assert item.task_type is TaskType.RETRIEVAL
        assert set(item.variants) == {Language.EN, Language.MSA, Language.GULF}


def test_slice_review_status_is_llm_qa_pending_native_review() -> None:
    # The README claims LLM QA only; native-speaker review has not happened.
    # When items are promoted to native_reviewed, update the README with it.
    for item in load_items(SLICE):
        assert item.review_status is ReviewStatus.LLM_QA


def test_slice_gold_ids_resolve_against_pinned_corpus() -> None:
    doc_ids = {doc.id for doc in load_documents(CORPUS)}
    for item in load_items(SLICE):
        unresolved = [g for g in item.gold.relevant_doc_ids if g not in doc_ids]
        assert not unresolved, f"{item.id}: unresolved gold ids {unresolved}"
