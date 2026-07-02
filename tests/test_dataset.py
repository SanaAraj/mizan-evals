"""Tests for the JSONL dataset loader, including the committed sample dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.dataset import DatasetError, load_items
from mizan.schemas import Language, TaskType

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = REPO_ROOT / "data" / "samples" / "eval_items.jsonl"

VALID_LINE = (
    '{"id": "d1", "task_type": "retrieval", '
    '"variants": {"en": {"query": "q"}}, '
    '"gold": {"relevant_doc_ids": ["doc1"]}}'
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_and_skips_blank_lines(tmp_path: Path) -> None:
    items = load_items(_write(tmp_path, f"\n{VALID_LINE}\n\n"))
    assert len(items) == 1
    assert items[0].id == "d1"


def test_missing_file_raises() -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_items("nope.jsonl")


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="empty"):
        load_items(_write(tmp_path, "\n\n"))


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=":2: invalid JSON"):
        load_items(_write(tmp_path, f"{VALID_LINE}\n{{ broken\n"))


def test_schema_violation_reports_line_number(tmp_path: Path) -> None:
    # Retrieval item without gold docs -> schema error, on line 1.
    bad = '{"id": "x", "task_type": "retrieval", "variants": {"en": {"query": "q"}}}'
    with pytest.raises(DatasetError, match=":1: invalid evaluation item"):
        load_items(_write(tmp_path, bad))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="duplicate item id"):
        load_items(_write(tmp_path, f"{VALID_LINE}\n{VALID_LINE}\n"))


# --- The committed sample dataset must always be valid and well-formed. ---


def test_sample_dataset_loads() -> None:
    items = load_items(SAMPLE_DATASET)
    assert len(items) == 10


def test_sample_items_are_marked_as_samples() -> None:
    for item in load_items(SAMPLE_DATASET):
        assert item.source == "sample"


def test_sample_items_are_parallel_across_all_three_languages() -> None:
    for item in load_items(SAMPLE_DATASET):
        assert set(item.variants) == {Language.EN, Language.MSA, Language.GULF}


def test_sample_dataset_covers_every_task_type() -> None:
    covered = {item.task_type for item in load_items(SAMPLE_DATASET)}
    assert covered == set(TaskType)
